# VisionReward

**VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation**
<!-- <div align="center">
🌟&nbsp;<a href="resources/WECHAT.md" target="_blank">Paper</a>&nbsp;·&nbsp;<a href="http://cogvlm2-online.cogviewai.cn:7861/" target="_blank">🤖 Demo</a>&nbsp;·&nbsp;<a href="http://cogvlm2-online.cogviewai.cn:7868/" target="_blank">📁 Dataset</a>
</div> -->
<!-- <div align="center">
🤗 HF repo :&nbsp;&nbsp; <a href="http://cogvlm2-online.cogviewai.cn:7861/" target="_blank">VisionReward-Image </a>&nbsp;&nbsp;<a href="http://cogvlm2-online.cogviewai.cn:7861/" target="_blank">VisionReward-Video</a>
</div> -->

<!-- ## Updates -->

## Overview
We present **VisionReward**, a general strategy to aligning visual generation models——both image and video generation——with human preferences through a fine-grainedand multi-dimensional framework. We decompose human preferences in images and videos into multiple dimensions,each represented by **a series of judgment questions**, linearly weighted and summed to an interpretable and accuratescore. 

To address the challenges of video quality assess-ment, we systematically analyze various dynamic features of videos, which helps VisionReward surpass VideoScore by 17.2% and achieve top performance for video preference prediction.
<div align="center">
<img src=asset/resource/overview.jpg width="90%"/> 
</div>

## Quick Start

### Set Up the Environment
Following the commands below to prepare the environment:
```
pip install -r requirements.txt
```
### VQA Example
Use the following code to perform a checklist query. You can view the available questions for images and videos in `VisionReward-Image/VisionReward_image_qa.txt` and `VisionReward-Video/VisionReward_video_qa.txt` respectively.
``` 
python inference-image.py --bf16 --question [[your_question]]
# input: image_path + prompt + question
# output: yes/no

python inference-video.py --question [[your_question]]
# input: video_path + prompt + question
# output: yes/no
```

### Using the model for scoring
Use the following code to score images/videos. The corresponding weights are in `VisionReward-Image/weight.json` and `VisionReward-Video/weight.json`.
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

<!-- ## Citation
If you find this work useful in your research, please consider citing:
```
@article{zhang2023visionreward,
  title={VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation},
``` -->