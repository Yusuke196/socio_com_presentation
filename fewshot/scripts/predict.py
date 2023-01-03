from itertools import islice
import subprocess
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import XGLMTokenizer, XGLMForCausalLM


def main():
    tokenizer = XGLMTokenizer.from_pretrained(
        "facebook/xglm-7.5B", cache_dir="cache"
    )
    model = XGLMForCausalLM.from_pretrained(
        "facebook/xglm-7.5B", cache_dir="cache"
    )

    device = "cuda:7"
    model = model.to(device)

    in_file_path = "data/prediction/all.csv"
    with open(in_file_path) as in_file:
        with open("data/prediction/fewshot_all.tsv", "w") as out_file:
            with tqdm(in_file, total=get_num_lines(in_file_path)) as pbar:
                # for line in islice(in_file, 10):
                for line in in_file:
                    x = "Tweet: " + line
                    # x = "Tweet: " + "ソフトバンクって打線は強いのに回線はなんで弱いの？"
                    samples = get_samples(x)
                    prompt = get_prompt(samples)
                    predict = eval(
                        prompt + "\n" + samples["choice1"],
                        prompt + "\n" + samples["choice2"],
                        tokenizer,
                        model,
                        device,
                    )
                    out_file.write(f"{line.rstrip()}\t{predict}\n")

                    pbar.update(1)


def get_num_lines(path):
    proc = subprocess.run(["wc", "-l", path], capture_output=True)
    res = int(proc.stdout.decode().split()[0])
    return res


def get_samples(x):
    samples = {
        "question": "次のツイートには、皮肉が含まれていますか？",
        "examples": [
            {
                "tweet": "Tweet: 紙ストローって「へ〜現代の技術なら紙のストローも実用に耐えるんだな〜」→耐えない　になるので本当にすごい",
                "answer": "Answer: 含まれる",
            },
            {
                "tweet": "Tweet: みんな家では変なダンスを踊って自分を保ってるんだと信じてる",
                "answer": "Answer: 含まれない",
            },
            {
                "tweet": "Tweet: 一度もお会いしたことがない人たちが、僕がSNSに投稿した画像や言葉を好き勝手に使い、テレビなどの公共放送で言いたい放題言っているようで、誠にご苦労様です。皆様がそのようなことに時間と労力を使うお仕事をされている中、僕は皆様のお役に立つための未来を考える仕事をしています。",
                "answer": "Answer: 含まれる",
            },
            {
                "tweet": "Tweet: 絶対に寝た方がいいのに、布団に入ってからが本当の自由という感じでテンションが上がってしまう",
                "answer": "Answer: 含まれない",
            },
            {
                "tweet": "Tweet: 五輪の新しい競技→踏んだり蹴ったり",
                "answer": "Answer: 含まれる",
            },
            {
                "tweet": "Tweet: ctrl + shift + escでタスクマネージャ起動はほんと為になってる\n\n#一番ためになったパソコン知識",
                "answer": "Answer: 含まれない",
            },
        ],
        "tweet": x,
        "choice1": "Answer: 含まれない",
        "choice2": "Answer: 含まれる",
    }
    return samples


def get_prompt(samples):
    paragraphs = [
        dct["tweet"] + "\n" + dct["answer"] for dct in samples["examples"]
    ]
    paragraphs.insert(0, samples["question"])
    paragraphs.append(samples["tweet"])
    prompt = "\n\n".join(paragraphs)
    return prompt


def eval(prompt1, prompt2, tokenizer, model, device):
    lprob1 = get_logprobs(prompt1, tokenizer, model, device).sum()
    lprob2 = get_logprobs(prompt2, tokenizer, model, device).sum()
    return 0 if lprob1 > lprob2 else 1


def get_logprobs(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logprobs = torch.gather(
        F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2)
    )
    return logprobs


if __name__ == "__main__":
    main()
