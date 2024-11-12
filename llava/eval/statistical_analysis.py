import argparse
import os
import time
import torch
from torch import nn
import json
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        question = line["conversations"][0]
        if question["from"] != "human":
            raise ValueError("Question should be from human")
        qs = question["value"].replace("<image>\n", "").replace("<image>", "")
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def calculate_flops(n=576, d=4096, m=11008, l=32):
    return (4 * n * d**2 + 2 * n**2 * d + 3 * n * m *d) * l


def rest_flops(token_num, layer_idx):
    flops = 0
    # output projection + FFN
    flops += (token_num * 4096**2 + 3 * token_num * 11008 * 4096)
    # flops for remaining layer
    flops += calculate_flops(token_num, l=31-layer_idx)
    return flops


def fit_model(model, questions, data_loader, mid_alpha, weightsum_self, weightsum_cross, origin_flops, reduction_ratio):
    flops_reach = False
    already_used_flops = 0
    left_vtoken_num_persample = {str(line["id"]): 576 for line in questions}
    residual_batch = {}
    position_ids_batch = {}
    scheme = {i: 0 for i in range(32)}

    for layer_idx, decoder_layer in enumerate(model.model.layers):
        flops_forcast = 0
        for vtoken_num in left_vtoken_num_persample.values():
            #                            qkv                    qk.T       *v                
            already_used_flops += (3 * vtoken_num * 4096**2 + 2 * vtoken_num**2 * 4096)
            flops_forcast += rest_flops(vtoken_num, layer_idx)
        if ((already_used_flops+flops_forcast)/len(left_vtoken_num_persample)) <= origin_flops*(1-reduction_ratio):
            print("Total FLOPs:", (already_used_flops+flops_forcast)/len(left_vtoken_num_persample))
            flops_reach = True
            break

        for (input_ids, image_tensor), line in zip(data_loader, questions):
            sample_idx = str(line["id"])

            if layer_idx == 0: 
                attention_mask = torch.ones_like(input_ids).to(device='cuda')
                input_ids = input_ids.to(device='cuda', non_blocking=True)
                image_tensor = image_tensor.to(device='cuda')
                input_ids, _, attention_mask, past_key_values, inputs_embeds, labels = model.prepare_inputs_labels_for_multimodal(input_ids, None, attention_mask, None, None, image_tensor.to(dtype=torch.float16))
                seq_length = inputs_embeds.shape[1]
                position_ids = torch.arange(
                    0, seq_length, dtype=torch.long, device=inputs_embeds.device
                ).unsqueeze(0).view(-1, seq_length)
            else:   
                inputs_embeds = residual_batch[sample_idx]
                if sample_idx in position_ids_batch:
                    position_ids = position_ids_batch[sample_idx]
                else:
                    seq_length = inputs_embeds.shape[1]
                    position_ids = torch.arange(
                        0, seq_length, dtype=torch.long, device=inputs_embeds.device
                    ).unsqueeze(0).view(-1, seq_length)
            
            batch_size, seq_length, _ = inputs_embeds.shape
            attention_mask = torch.ones(1, seq_length)
            attention_mask = model.model._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, 0
            )
            
            hidden_states = inputs_embeds

            residual_batch[sample_idx] = hidden_states

            hidden_states = decoder_layer.input_layernorm(hidden_states)
            bsz, q_len, _ = hidden_states.size()
            self_attn = decoder_layer.self_attn
            query_states = self_attn.q_proj(hidden_states)
            key_states = self_attn.k_proj(hidden_states)
            value_states = self_attn.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)

            cos, sin = self_attn.rotary_emb(value_states, seq_len=position_ids.max()+1)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
            value_states = repeat_kv(value_states, self_attn.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self_attn.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            
            attn_weights_headmax = attn_weights.max(dim=1).values.squeeze()
            left_vtoken_num = left_vtoken_num_persample[sample_idx]
            attn_weights_self = attn_weights_headmax[35:35+left_vtoken_num, 35:35+left_vtoken_num].sum(dim=0) / left_vtoken_num
            attn_weights_cross = attn_weights_headmax[35+left_vtoken_num:, 35:35+left_vtoken_num].sum(dim=0) / (attn_weights_headmax.shape[0] - 35 - left_vtoken_num)

            threshold_self = weightsum_self[sample_idx][layer_idx] * mid_alpha
            threshold_cross = weightsum_cross[sample_idx][layer_idx] * mid_alpha

            # Sort by self and cross weights in ascending order
            sorted_self_indices = torch.argsort(attn_weights_self)
            sorted_cross_indices = torch.argsort(attn_weights_cross)

            del_self_idx = []
            del_cross_idx = []

            self_sum = attn_weights_self.sum()
            cross_sum = attn_weights_cross.sum()

            for i in range(len(sorted_self_indices)):
                if (self_sum - attn_weights_self[sorted_self_indices[:i]].sum()) < (weightsum_self[sample_idx][layer_idx] - threshold_self):
                    del_num_self = max(i-1, 0)
                    break
            del_self_idx = sorted_self_indices[:del_num_self]

            for i in range(len(sorted_cross_indices)): 
                if (cross_sum - attn_weights_cross[sorted_cross_indices[:i]].sum()) < (weightsum_cross[sample_idx][layer_idx] - threshold_cross):
                    del_num_cross = max(i-1, 0)
                    break
            del_cross_idx = sorted_cross_indices[:del_num_cross]
            
            # Intersection
            combined = torch.concat([del_self_idx, del_cross_idx])
            combined_val, counts = combined.unique(return_counts=True, sorted=False)
            topk_intersec = combined_val[counts>1]
            scheme[layer_idx] += topk_intersec.shape[0]
            if topk_intersec.shape[0] > 0:
                mask = torch.ones(attn_output.shape[2], dtype=torch.bool, device=attn_weights.device)
                mask[topk_intersec + 35] = False

                attn_output = attn_output[:, :, mask, :]

                residual_batch[sample_idx] = residual_batch[sample_idx][:, mask, :]
                left_vtoken_num_persample[sample_idx] -= (~mask).sum().item()

                position_ids = position_ids[:, mask]
                position_ids_batch[sample_idx] = position_ids

            q_len = attn_output.shape[2]
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self_attn.hidden_size)
            attn_output = self_attn.o_proj(attn_output)
            
            # + residual
            residual = residual_batch[sample_idx]
            hidden_states = residual + attn_output

            # FFN   
            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            residual_batch[sample_idx] = hidden_states

            # update already_use_flops
            already_used_flops += left_vtoken_num_persample[sample_idx] * 4096**2 + 3 * left_vtoken_num_persample[sample_idx] * 11008 * 4096

    # check flops budget
    if not flops_reach and already_used_flops/len(left_vtoken_num_persample) <= origin_flops*(1-reduction_ratio):
        print("Total FLOPs:", already_used_flops/len(left_vtoken_num_persample))
        flops_reach = True

    scheme = {i: round(scheme[i]//len(questions)) for i in range(32)}
    return flops_reach, scheme


def compute_attention_weights(model, questions, data_loader):
    system_length = 35
    visual_length = 576

    weightsum_self = {}
    weightsum_cross = {}

    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):

        sample_idx = str(line["id"])
        weightsum_self[sample_idx] = {}
        weightsum_cross[sample_idx] = {}
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        attention_mask = torch.ones_like(input_ids)

        input_ids, _, attention_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(input_ids, None, attention_mask, None, None, image_tensor)
        
        with torch.inference_mode():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True)
            for i in range(32):
                attn_weights = outputs.attentions[i].max(dim=1).values.squeeze()
                attn_self = attn_weights[system_length:system_length+visual_length, system_length:system_length+visual_length]
                attn_cross = attn_weights[system_length+visual_length:, system_length:system_length+visual_length]
                weightsum_self[sample_idx][i] = attn_self.sum().item() / visual_length
                weightsum_cross[sample_idx][i] = attn_cross.sum().item() / (attn_weights.shape[0] - system_length - visual_length)
    
    return weightsum_self, weightsum_cross


def statistical_analysis(args):
    # Initialize model and data loader
    print("Initializing...")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path) + "-fitprune"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.eval()
    model.config.use_fitprune = False
    model.config.reduction_ratio = 0
    
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        
    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    start_time = time.time()

    # Compute attention weights
    print("Computing...")
    weightsum_self, weightsum_cross = compute_attention_weights(model, questions, data_loader)

    # Fit attention weights distribution and calculate scheme
    print("Fitting...")
    reduction_ratio = args.reduction_ratio / 100
    origin_flops = calculate_flops()
    print("Original TFLOPs:", origin_flops/1e12)

    max_alpha, min_alpha = 1.0, 0.0
    while max_alpha - min_alpha > args.epsilon:
        mid_alpha = (min_alpha + max_alpha) / 2
        print(f"max_alpha: {max_alpha}, min_alpha: {min_alpha}, mid_alpha: {mid_alpha}")
        flops_reach, scheme = fit_model(model, questions, data_loader, mid_alpha, weightsum_self, weightsum_cross, origin_flops, reduction_ratio)
        if flops_reach:
            max_alpha = mid_alpha
        else:
            min_alpha = mid_alpha
    
    print("Time elapsed:", time.time() - start_time)
    
    print("scheme: ", scheme)
    print("rest visual tokens: ", 576-sum(scheme.values()))
    temp_visual_tokens = 576
    rest_visual_tokens = []
    for i in range(32):
        temp_visual_tokens -= scheme[i]
        rest_visual_tokens.append(temp_visual_tokens)
    print("avg visual tokens: ", sum(rest_visual_tokens)/32)

    pruned_flops = 0
    visual_tokens = 576
    for i in range(32):
        pruned_flops += 3 * visual_tokens * 4096**2 + 2 * visual_tokens**2 * 4096
        visual_tokens -= scheme[i]
        pruned_flops += visual_tokens * 4096**2 + 3 * visual_tokens * 11008 * 4096
    print("Pruned TFLOPs:", pruned_flops/1e12)
    print("FLOPs reduction: ", (origin_flops - pruned_flops) / origin_flops * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/bn/bes-nas-zqz-lq-v6arnold6/mlx/users/zhangqizhe/huggingface/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./playground/data/train")
    parser.add_argument("--question-file", type=str, default="./playground/data/train/llava_v1_5_mix665.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--reduction-ratio", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    with torch.no_grad():
        statistical_analysis(args)
