# Speaker Verification Demo

This code is modified based on https://github.com/clovaai/voxceleb_trainer 

The pretrain model used here is also the model that repositories provided.

Speaker verification is to judge if two utterance belong to the same speaker based on the each person's unique voiceprint 

## How to use:
### Input: 
  
  Put the wav files into the data folder, change the name in run.sh file. One is for enrollment_audio, another is for test_audio.

### Commend:

```
bash run.sh
```

### Output: 
  
  The score is the speaker verification score between two utterance, the score is higher, the utterance tends to come from the same speaker (The highest is 0.0)


## Other issue:

- Is there any requirement for the wav file ?
- 
No, any sampling rate, single/dual channel is fine. 

- Requirement
- 
Pytorch, soundfile package

- Threshold setting
- 
It depend on the dataset. Just for suggestion: usually the score between 0 to -1.0 can be viewed as the same speaker, score small than -1.0 can be viewed as different speaker.

- Reference
- 
Please check their paper for more details:

	```
	@inproceedings{chung2020in,
	  title={In defence of metric learning for speaker recognition},
	  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
	  booktitle={Interspeech},
	  year={2020}
	}
	```
	```
	@article{heo2020clova,
	  title={Clova baseline system for the {VoxCeleb} Speaker Recognition Challenge 2020},
	  author={Heo, Hee Soo and Lee, Bong-Jin and Huh, Jaesung and Chung, Joon Son},
	  journal={arXiv preprint arXiv:2009.14153},
	  year={2020}
	}
	```
