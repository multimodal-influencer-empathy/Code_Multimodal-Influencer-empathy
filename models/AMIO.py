import torch.nn as nn
from models.subNets import AlignSubNet

from models.A1_Graph_MFN_L import A1_Graph_MFN_L
from models.A2_Graph_MFN_A import A2_Graph_MFN_A
from models.A3_Graph_MFN_I import A3_Graph_MFN_I
from models.A4_Graph_MFN_LA import A4_Graph_MFN_LA
from models.A5_Graph_MFN_LI import A5_Graph_MFN_LI
from models.A6_Graph_MFN_AI import A6_Graph_MFN_AI
from models.A7_Graph_MFN import A7_Graph_MFN
from models.A8_Graph_MFN_noM import A8_Graph_MFN_noM
from models.A9_Graph_MFN_noG import A9_Graph_MFN_noG
from models.A10_Graph_MFN_noW import A10_Graph_MFN_noW
from models.A11_EF_LSTM import A11_EF_LSTM
from models.A12_TFN import A12_TFN
from models.A13_LMF import A13_LMF
from models.A14_MFN import A14_MFN

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.MODEL_MAP = {
              'A1_Graph_MFN_L': A1_Graph_MFN_L,
            'A2_Graph_MFN_A': A2_Graph_MFN_A,
            'A3_Graph_MFN_I': A3_Graph_MFN_I,
            'A4_Graph_MFN_LA': A4_Graph_MFN_LA,
            'A5_Graph_MFN_LI': A5_Graph_MFN_LI,
            'A6_Graph_MFN_AI': A6_Graph_MFN_AI,
            'A7_Graph_MFN': A7_Graph_MFN,
            'A8_Graph_MFN_noM': A8_Graph_MFN_noM,
            'A9_Graph_MFN_noG': A9_Graph_MFN_noG,
            'A10_Graph_MFN_noW': A10_Graph_MFN_noW,
            'A11_EF_LSTM': A11_EF_LSTM,
            'A12_TFN': A12_TFN,
            'A13_LMF': A13_LMF,
            'A14_MFN': A14_MFN
        }
        self.need_model_aligned = args['need_model_aligned']
        if(self.need_model_aligned):
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()
        lastModel = self.MODEL_MAP[args['model_name']]
        print("Model:", args['model_name'])
        print("\n The args in the model is as below: \n", args)
        self.model_name = args['model_name']
        self.Model = lastModel(args)


    def forward(self, text_x, audio_x, vision_x, *args,**kwargs):
        if (self.need_model_aligned):
            text_x, audio_x, vision_x = self.alignNet(text_x, audio_x, vision_x)
        return self.Model(text_x, audio_x, vision_x,  *args, **kwargs)
