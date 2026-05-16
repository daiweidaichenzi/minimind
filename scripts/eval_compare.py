"""
GQA vs MLA 模型对比评估脚本
- Perplexity（在 SFT 数据子集上）
- 指定题目人工对比
"""
import torch, json, argparse, random, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_minimind_mla import MiniMindMLAConfig, MiniMindMLAForCausalLM
from transformers import AutoTokenizer

TEST_QUESTIONS = [
    "请用Python写一个计算斐波那契数列的函数",
    "解释一下机器学习和深度学习的区别",
    "推荐三本经典的科幻小说，并说明理由",
    "如果我想在北京开一家小餐馆，需要注意哪些方面",
    "请解释什么是光合作用",
    "比较一下燃油车和电动车的优缺点",
    "写一首关于秋天的五言绝句",
    "如何给孩子解释为什么天空是蓝色的",
    "请用简单的语言说明什么是API",
    "介绍一道你最喜欢的菜的做法",
    "什么是区块链技术，它可以应用在哪些领域",
    "如何有效地管理时间",
    "请翻译：Artificial intelligence is transforming every industry",
    "三个开关分别控制三个灯泡，你只能进房间一次，如何确定对应关系",
    "请用200字以内介绍中国的四大发明",
]


def load_model(weight_path, config, device):
    """加载模型权重"""
    if isinstance(config, MiniMindMLAConfig):
        model = MiniMindMLAForCausalLM(config)
    else:
        model = MiniMindForCausalLM(config)
    model = model.to(device)
    if weight_path and os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"  加载权重: {weight_path}")
    model.eval()
    return model


def compute_ppl(model, tokenizer, data_path, device, max_samples=200, max_length=512):
    """在数据集子集上计算 Perplexity"""
    print(f"\n  数据: {data_path} (最多 {max_samples} 条, max_len={max_length})")

    try:
        with open(data_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"  [WARN] 数据文件不存在，跳过 PPL")
        return None

    random.shuffle(lines)
    total_loss = 0.0
    total_tokens = 0
    count = 0

    for line in lines[:max_samples]:
        item = json.loads(line)
        # SFT 格式：有 conversations 字段
        if "conversations" in item:
            text = tokenizer.apply_chat_template(item["conversations"], tokenize=False, add_generation_prompt=False)
        else:
            text = item.get("text", "")
        if not text.strip():
            continue

        tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device)
        with torch.no_grad():
            output = model(input_ids, labels=input_ids)
            loss = output.loss
            if loss is not None:
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        count += 1

    if total_tokens == 0:
        return None
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"  处理 {count} 条, 总 token {total_tokens}, loss={avg_loss:.4f}, ppl={ppl:.2f}")
    return ppl


@torch.inference_mode()
def generate_answer(model, tokenizer, prompt, device, max_new_tokens=512):
    """生成回答"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    output = model.generate(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    response = tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gqa_weight", type=str, default="../out/full_sft_768.pth")
    parser.add_argument("--mla_weight", type=str, default="../out/full_sft_768_mla.pth")
    parser.add_argument("--tokenizer_path", type=str, default="../model")
    parser.add_argument("--sft_data", type=str, default="../dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--ppl_samples", type=int, default=200, help="PPL 采样条数")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--skip_ppl", action="store_true", help="跳过 PPL 计算")
    parser.add_argument("--skip_qna", action="store_true", help="跳过问答对比")
    parser.add_argument("--only_ppl", action="store_true", help="只算 PPL")
    args = parser.parse_args()

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print("=" * 80)
    print("  GQA vs MLA 模型对比评估")
    print("=" * 80)

    # ── 1. 加载模型 ──
    print("\n[1] 加载 GQA 模型...")
    gqa_config = MiniMindConfig()
    gqa_model = load_model(args.gqa_weight, gqa_config, device)

    print("\n[2] 加载 MLA 模型...")
    mla_config = MiniMindMLAConfig()
    mla_model = load_model(args.mla_weight, mla_config, device)

    # ── 2. PPL ──
    if not args.skip_ppl:
        print("\n" + "=" * 80)
        print("  Perplexity 对比")
        print("=" * 80)
        print("\n[GQA]")
        gqa_ppl = compute_ppl(gqa_model, tokenizer, args.sft_data, device, max_samples=args.ppl_samples)
        print("\n[MLA]")
        mla_ppl = compute_ppl(mla_model, tokenizer, args.sft_data, device, max_samples=args.ppl_samples)

        if gqa_ppl and mla_ppl:
            better = "GQA" if gqa_ppl < mla_ppl else "MLA"
            print(f"\n  结果: GQA PPL={gqa_ppl:.2f}  MLA PPL={mla_ppl:.2f}  →  {better} 更低({abs(gqa_ppl - mla_ppl):.2f})")

    if args.only_ppl:
        return

    # ── 3. 问答对比 ──
    if not args.skip_qna:
        print("\n" + "=" * 80)
        print("  问答对比")
        print("=" * 80)

        for i, question in enumerate(TEST_QUESTIONS, 1):
            print(f"\n{'─' * 80}")
            print(f"  [{i}/{len(TEST_QUESTIONS)}] {question}")
            print(f"{'─' * 80}")

            print(f"\n  [GQA]:")
            gqa_answer = generate_answer(gqa_model, tokenizer, question, device, max_new_tokens=args.max_new_tokens)
            for line in gqa_answer.split('\n'):
                print(f"    {line}")

            print(f"\n  [MLA]:")
            mla_answer = generate_answer(mla_model, tokenizer, question, device, max_new_tokens=args.max_new_tokens)
            for line in mla_answer.split('\n'):
                print(f"    {line}")

    print("\n" + "=" * 80)
    print("  评估完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
