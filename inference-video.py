import io
import json
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm

MODEL_PATH = "THUDM/VisionReward-Video"
QUESTIONS_PATH = "VisionReward_Video/VisionReward_video_qa_select.txt"
WEIGHT_PATH = "VisionReward_Video/weight.json"

with open(QUESTIONS_PATH, 'r') as f:
    questions = f.readlines()

with open(WEIGHT_PATH, 'r') as f:
    weight = json.load(f)
    weight = np.array(weight)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    # padding_side="left"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True
).eval().to(DEVICE)


def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data

def inference(video_path, query, temperature=0.1):

    video_data = open(video_path, 'rb').read()              
    strategy = 'chat'
    video = load_video(video_data, strategy=strategy)
    
    history = []

    yes_token_id = tokenizer.encode("Yes")[0]

    inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer,
            query=query,
            images=[video],
            history=history,
            template_version=strategy
        )
    inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
        }
    gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": False,
            "top_p": 0.1,
            "temperature": temperature,
        }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]]
    
    return tokenizer.decode(outputs[0])

def score(video_path, prompt) -> float:
    queries = [question.replace('[[prompt]]', prompt) for question in questions]
    answers = []
    for query in tqdm(queries, 'scoring video'):
        answer = inference(video_path, query)
        answers.append(answer)
    answers = np.array([1 if answer == 'yes' else -1 for answer in answers])
    return np.mean(answers * weight).item()

def compare_two_videos(video_path1, video_path2, prompt) -> bool:
    queries = [question.replace('[[prompt]]', prompt) for question in questions]

    answers1, answers2 = [], []
    for query in tqdm(queries, 'scoring video 1'):
        answer = inference(video_path1, query)
        answers1.append(answer)
    answers1 = np.array([1 if answer == 'yes' else -1 for answer in answers1])
    for query in tqdm(queries, 'scoring video 2'):
        answer = inference(video_path2, query)
        answers2.append(answer)
    answers2 = np.array([1 if answer == 'yes' else -1 for answer in answers2])

    diff = answers1 - answers2
    
    return np.sum(diff * weight).item() > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")

    parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
    parser.add_argument('--question', type=str, help='Question to be answered', default='Is there a man in the video?')
    parser.add_argument('--score', help='Whether to output the score', default=False, action='store_true')
    parser.add_argument('--compare', help='Whether to compare two videos', default=False, action='store_true')

    args = parser.parse_args()

    video1 = './asset/test/test1.mp4'
    video2 = './asset/test/test2.mp4'
    prompt = 'Multiple elephants inhabit a surreal and dystopian urban landscape where towering trees emerge from the cracked city streets, their roots intertwining with skyscrapers, under an eerie, blood-red sky that looms overhead.'

    if args.score:
        print('Score mode')
        print(score(video1, prompt))
    elif args.compare:
        print('Compare mode')
        print('video1 > video2: ' if compare_two_videos(video1, video2, prompt) else 'video1 < video2')
    else:
        print('Question mode')
        print(inference(video1, args.question))
