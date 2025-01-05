# VisionReward

<p align="center">
   📃 <a href="https://arxiv.org/abs/2412.21059" target="_blank">Paper</a> • 🖼 <a href="https://github.com/THUDM/VisionReward" target="_blank">Dataset (Coming soon) </a> • 🤗 <a href="https://huggingface.co/THUDM/VisionReward-Video" target="_blank">HF Repo</a> • 🌐 <a href="https://zhuanlan.zhihu.com/p/16481080277" target="_blank">中文博客</a> <br>
</p>

**VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation**

VisionReward is a fine-grained and multi-dimensional reward model. We decompose human preferences in images and videos into multiple dimensions, each represented by a series of judgment questions, linearly weighted and summed to an interpretable and accurate score. To address the challenges of video quality assessment, we systematically analyze various dynamic features of videos, which helps VisionReward surpass VideoScore by 17.2% and achieve top performance for video preference prediction.

<div align="center">
<img src=asset/resource/overview.jpg width="90%"/> 
</div>

## Quick Start

### Set Up the Environment
Following the commands below to prepare the environment:
```
pip install -r requirements.txt
```

### Download the model
You can download the pre-trained **VisionReward** models for images and videos from the following Hugging Face repositories:

- **Image Reward Model**: [https://huggingface.co/THUDM/VisionReward-Image](https://huggingface.co/THUDM/VisionReward-Image)
- **Video Reward Model**: [https://huggingface.co/THUDM/VisionReward-Video](https://huggingface.co/THUDM/VisionReward-Video)

### VQA Example
Use the following code to perform a checklist query. You can view the available questions for images and videos in `VisionReward_Image/VisionReward_image_qa.txt` and `VisionReward_Video/VisionReward_video_qa.txt` respectively.
``` 
python inference-image.py --bf16 --question [[your_question]]
# input: image_path + prompt + question
# output: yes/no

python inference-video.py --question [[your_question]]
# input: video_path + prompt + question
# output: yes/no
```

### Using the model for scoring
Use the following code to score images/videos. The corresponding weights are in `VisionReward_Image/weight.json` and `VisionReward_Video/weight.json`.
``` 
python inference-image.py --bf16 --score 
# input: image_path + prompt
# output: score

python inference-video.py --score
# input: video_path + prompt
# output: score
```

### Using the model for comparing two videos
Use the following code to compare two videos. The corresponding weights are in `VisionReward-Video/weight.json`.
```
python inference-video.py --compare
# input: video_path1 + video_path2 + prompt
# output: better_video
```

## Demos of VisionReward

<p align="center">
    <img src="asset/resource/VisionReward_demo.jpg" width="700px">
</p>

## Citation

```
@misc{xu2024visionrewardfinegrainedmultidimensionalhuman,
      title={VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation}, 
      author={Jiazheng Xu and Yu Huang and Jiale Cheng and Yuanming Yang and Jiajun Xu and Yuan Wang and Wenbo Duan and Shen Yang and Qunlin Jin and Shurun Li and Jiayan Teng and Zhuoyi Yang and Wendi Zheng and Xiao Liu and Ming Ding and Xiaohan Zhang and Xiaotao Gu and Shiyu Huang and Minlie Huang and Jie Tang and Yuxiao Dong},
      year={2024},
      eprint={2412.21059},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.21059}, 
}
```
