import argparse, numpy, soundfile, importlib, warnings, torch, os, glob, pandas
import os.path as osp
from .models.ResNetSE34V2 import MainModel

CUR_DIR = osp.dirname(osp.abspath(__file__))

class SpeakerRecognizer():
    def __init__(self) -> None:
        model = MainModel()
        self.model = self.loadPretrain(model, osp.join(CUR_DIR, 'models/pretrain.model'))
        self.model.eval()

        # audio database
        self.enroll_audios = glob.glob(osp.join(CUR_DIR, 'data/enroll/*.wav'))
        self.utt_enroll_dict = {}
        for audio in self.enroll_audios:
            self.utt_enroll_dict[audio] = self.loadWAV(audio)

    def loadPretrain(self, model, pretrain_model):
        self_state = model.state_dict()
        loaded_state = torch.load(pretrain_model, map_location="cpu")
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("__S__.", "")
                if name not in self_state:
                    continue
            self_state[name].copy_(param)
        self_state = model.state_dict()
        return model

    def loadWAV(self, filename):
        audio, sr = soundfile.read(filename)
        if len(audio.shape) == 2: # dual channel will select the first channel
            audio = audio[:,0]

        feat = numpy.stack([audio],axis=0).astype(numpy.float)
        feat = torch.FloatTensor(feat)
        return feat

    def remember_new_speaker(self, filename):
        speaker_name = osp.splitext(filename)[0]
        self.utt_enroll_dict[speaker_name] = self.loadWAV(osp.join(CUR_DIR, "../audio_file/{}".format(filename)))
        print("[SpeakerRecognizer]: remembered new speaker {}!!".format(speaker_name))
        return True

    def recognize_audio(self, audio_feat):
        audio_feat = torch.FloatTensor(audio_feat)
        score_matrix = {}
        feat_test = self.model(audio_feat).detach()
        feat_test = torch.nn.functional.normalize(feat_test, p=2, dim=1)
        for enroll_audio in self.utt_enroll_dict.keys():
            feat_enroll = self.model(self.utt_enroll_dict[enroll_audio]).detach()
            feat_enroll = torch.nn.functional.normalize(feat_enroll, p=2, dim=1)
            score = [float(numpy.round(- torch.nn.functional.pairwise_distance(feat_enroll.unsqueeze(-1), feat_test.unsqueeze(-1).transpose(0,2)).detach().numpy(), 4))]
            score_matrix[enroll_audio.split('/')[-1].split('.')[0]] = score

        score_sorted = sorted(score_matrix.items(), key=lambda item: item[1], reverse = True)
        top_speaker, score = score_sorted[0]
        score = score[0]
        print("[SpeakerRecognizer]: top_speaker: {}, score: {}".format(top_speaker, score))

        return top_speaker, score

    def recognize_file(self, filename):
        audio_feat = self.loadWAV(osp.join(CUR_DIR, "../audio_file/{}".format(filename)))
        score_matrix = {}
        feat_test = self.model(audio_feat).detach()
        feat_test = torch.nn.functional.normalize(feat_test, p=2, dim=1)
        for enroll_audio in self.utt_enroll_dict.keys():
            feat_enroll = self.model(self.utt_enroll_dict[enroll_audio]).detach()
            feat_enroll = torch.nn.functional.normalize(feat_enroll, p=2, dim=1)
            score = [float(numpy.round(- torch.nn.functional.pairwise_distance(feat_enroll.unsqueeze(-1), feat_test.unsqueeze(-1).transpose(0,2)).detach().numpy(), 4))]
            score_matrix[enroll_audio.split('/')[-1].split('.')[0]] = score

        score_sorted = sorted(score_matrix.items(), key=lambda item: item[1], reverse = True)
        top_speaker, score = score_sorted[0]
        score = score[0]
        print("[SpeakerRecognizer]: top_speaker: {}, score: {}".format(top_speaker, score))

        return top_speaker, score

if __name__ == '__main__':
    recog = SpeakerRecognizer()
    speaker, score = recog.recognize(osp.join(CUR_DIR, 'data/test/A1.wav'))
    print(speaker)