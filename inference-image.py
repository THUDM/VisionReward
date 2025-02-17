# -*- encoding: utf-8 -*-
import os, sys
import json
import argparse
import torch
from tqdm import tqdm  # Import tqdm for the progress bar
from sat.model.mixins import CachedAutoregressiveMixin
from sat.model import AutoModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.utils import llama3_tokenizer
from utils.models import CogVLMModel
from utils.models import VisualLlamaEVA
from io import BytesIO
import pandas as pd
from PIL import Image
import numpy as np
from VisionReward_Image.t2v_metrics.vqascore import VQAScore

MASK_INDICES = [0, 1, 2]      # Indices of mask features in original list
MASK_FEATURE_MAP = {
    0: [22, 23, 24, 28, 29],      # 'body(mask)' masks related features 'body correct' & 'harmfulness'
    1: [25, 26],                  # 'face(mask)' masks related features 'face'
    2: [27],                      # 'hands(mask)' masks related features 'hands'
}

def cal_score(args,image_path,prompt,model,text_processor_infer,image_processor):
    with open(args.ques_file, 'r') as file:
        ques_data = [line.strip() for line in file]
    with open(args.weight_file, 'r') as file2:
        weight_data = json.load(file2)
    wegiht = weight_data['coef']
    intercept = weight_data['intercept']
    answer_list = []
    alignment_score = VQAScore(model='clip-flant5-xxl') # our recommended scoring model
    alignment = alignment_score(images=[image_path], texts=[prompt])[0][0].cpu().item() 
    for ques in tqdm(ques_data, f'scoring image:{image_path}'):
        try:
            response, _, _ = chat(
                image_path=image_path,
                image = None,
                model=model,
                text_processor=text_processor_infer,
                img_processor=image_processor,
                query=ques,
                max_length=args.max_length,
                top_p=args.top_p,
                temperature=args.temperature,
                top_k=args.top_k,
                invalid_slices=text_processor_infer.invalid_slices,
                args=args
            )
            answer_list.append(response)
        except Exception as e:
            answer_list.append(None)
            print(f"Error processing {ques}: {str(e)}")
    reward = [(1 if ans =='yes<|end_of_text|>' else -1 ) for ans in answer_list]
    # add mask
    for mask_index, feature_indices in MASK_FEATURE_MAP.items():
        for feature_index in feature_indices:
            reward[feature_index] *= (int)(reward[mask_index] > 0)
    reward_filtered = [v for i, v in enumerate(reward) if i not in MASK_INDICES]
    final_reward = [alignment] + reward_filtered
    score = np.dot(final_reward, wegiht) + intercept
    return score[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=3328, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=0.8, help='temperature for sampling')
    parser.add_argument("--version", type=str, default="vqa", choices=['chat', 'vqa', 'chat_old', 'base'], help='version of language process')
    parser.add_argument("--from_pretrained", type=str, default="THUDM/VisionReward-Image", help='pretrained ckpt')  # You need to first download the model from https://huggingface.co/THUDM/VisionReward-Image and then refer to its README to extract the checkpoint.
    parser.add_argument("--tokenizer_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true", help="Use fp16 precision")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision")
    parser.add_argument("--stream_chat", action="store_true")
    parser.add_argument("--ques_file", type=str, default="VisionReward_Image/VisionReward_image_qa_select.txt", help="Path to the meta question file")
    parser.add_argument("--weight_file", type=str, default="VisionRewardImage/weight_select.json", help="Path to the weight file")
    parser.add_argument('--question', type=str, help='Question to be answered', default='Is the image clear?')
    parser.add_argument('--score', help='Whether to output the score', default=False, action='store_true')
    args = parser.parse_args()

    # Initialize model
    model, model_args = VisualLlamaEVA.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
            deepspeed=None,
            local_rank=0,
            rank=0,
            world_size=1,
            model_parallel_size=1,
            mode='inference',
            skip_init=True,
            use_gpu_initialization=True,
            device='cuda',
            **vars(args)
        )
    )
    model = model.eval()
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    tokenizer = llama3_tokenizer(args.tokenizer_path, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)
    
    # Set input
    image_path1 = "asset/test/test1.jpg"
    image_path2 = "asset/test/test2.jpg"
    prompt = "A close-up portrait of a beautiful girl with an autumn leaves headdress and melting wax."
    
    with torch.no_grad():
        if args.score:
            score = cal_score(args,image_path1,prompt,model,text_processor_infer,image_processor)
            print(f"score: {score}")
        else:
            ques = args.question
            response, _, _ = chat(
                image_path=image_path1,
                image = None,
                model=model,
                text_processor=text_processor_infer,
                img_processor=image_processor,
                query=ques,
                max_length=args.max_length,
                top_p=args.top_p,
                temperature=args.temperature,
                top_k=args.top_k,
                invalid_slices=text_processor_infer.invalid_slices,
                args=args
            )
            print(f"response:{response}")
               

if __name__ == "__main__":
    main()
