#import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



def get_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def step_forward(model, tokenizer, prompt, decoding=True, k_indices=5):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(model.device)
    input_ids = inputs['input_ids']
    with torch.no_grad():
        outputs = model(input_ids)
        tl_pair = []
        lm_head = model.lm_head
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            norm = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            norm = model.transformer.ln_f
        else:
            raise ValueError(f"Incorrect Model")
        for i, r in enumerate(outputs.hidden_states):
            layer_logits = []
            layer_output = norm(r)
            logits = lm_head(layer_output)
            next_token_logits = logits[:, -1, :]
            top_logits_k = k_indices
            top_values, top_indices = torch.topk(next_token_logits, top_logits_k, dim=-1)
            decoded_texts = [tokenizer.decode([idx], skip_special_tokens=False) for idx in top_indices.squeeze().tolist()]
            top_values = top_values.detach().cpu()
            if decoding:
                for value, token in zip(top_values.squeeze().tolist(), decoded_texts):
                    layer_logits.append([token, value])
            else:
                for value, token in zip(top_values.squeeze().tolist(), top_indices.squeeze().tolist()):
                    layer_logits.append([token, value])
            tl_pair.append(layer_logits)
    res_hidden_states = []
    for _ in outputs.hidden_states:
        res_hidden_states.append(_.detach().cpu().numpy())
    return res_hidden_states, tl_pair

def calculate_last_layer_entropy(model, tokenizer, prompt):
    # 将提示符编码为输入张量
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(model.device)
    input_ids = inputs['input_ids']

    with torch.no_grad():
        # 前向传播得到模型输出
        outputs = model(input_ids)

        # 获取最后一层的 hidden states
        last_hidden_state = outputs.hidden_states[-1]

        # 对最后一层应用 norm 层（根据模型结构）
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            norm = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            norm = model.transformer.ln_f
        else:
            raise ValueError(f"Incorrect Model")

        # 计算最后一层的 logits
        layer_output = norm(last_hidden_state)
        logits = model.lm_head(layer_output)  # 获取最后一层的 logits
        next_token_logits = logits[:, -1, :]  # 获取下一个 token 的 logits

        # 计算最后一层 softmax 概率分布
        probs = torch.softmax(next_token_logits, dim=-1)

        # 计算预测熵（预测不确定性）
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

    return entropy.item()  # 返回预测熵的值