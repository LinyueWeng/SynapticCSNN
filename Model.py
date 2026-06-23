import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from Layers import CsnnLayer,SnnPooling
from sklearn.svm import LinearSVC
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys
from config import *

#device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_local='cpu'

class CSNN_Layerwise:

    def __init__(self,synapse_model="Ferroelectric",device=device_local,v=1.0,sfp=1.033,sfd=1.30):
        self.device = device
        self.timesteps = TIMESTEPS
        self.v_rest = 0
        self.v_thresh = 1
        self.v_reset = -1
        self.r_inhib = 3
        self.f_dep = 2
        self.lr = 0.01
        self.sfd=sfd
        self.synapse_model=synapse_model

        self.conv1 = CsnnLayer(self.input_shape(1),128,tau=1.0,v=v,sfp=sfp,sfd=self.sfd,synapse_model=self.synapse_model,kernel_size=7,stride=1,padding=3,lr=self.lr,f_dep=self.f_dep,v_rest=self.v_rest,v_thresh=self.v_thresh,v_reset=self.v_reset,r_inhib=self.r_inhib,device=self.device,timesteps=self.timesteps,n_winners=7)
        self.pool1 = SnnPooling((self.timesteps,self.conv1.output_channels,self.conv1.output_h,self.conv1.output_w),kernel_size=3,stride=3,padding=0,v_thresh=1,timesteps=self.timesteps,device=self.device)
        if hasattr(self.conv1, 'weight'):
            self.conv1.weight = self.conv1.weight.to(self.device)

        # self.conv2 = CsnnLayer((self.timesteps,self.conv1.output_channels,self.conv1.output_h,self.conv1.output_w),96,v=v,sfp=sfp,sfd=self.sfd,synapse_model=self.synapse_model,kernel_size=3,stride=1,padding=0,lr=self.lr,f_dep=self.f_dep,v_rest=self.v_rest,v_thresh=self.v_thresh*1.2,v_reset=self.v_reset,r_inhib=self.r_inhib,device=self.device,timesteps=self.timesteps)
        # self.pool2 = SnnPooling((self.timesteps,self.conv2.output_channels,self.conv2.output_h,self.conv2.output_w),kernel_size=2,stride=2,padding=0,v_thresh=1,timesteps=self.timesteps,device=self.device)
        if hasattr(self.conv1, 'weight'):
            self.conv1.weight = self.conv1.weight.to(self.device)

        self.svm_classifier = LinearSVC(C=0.005)
        self.is_svm_trained = False



    def input_shape(self,channels):
        ##midify the size of each image here!##
        return 1,channels,28,28

    def forward(self,image_tensor,is_training=False,lr=0.01):
        if image_tensor.device != self.device:
            image_tensor = image_tensor.to(self.device)

        potential,spike=self.conv1.ttfs_inputlayer(image_tensor)
        potential,spike=self.conv1(potential,spike,is_training=is_training)
        #self.see_potential_evolve(potential)
        potential,spike=self.pool1(potential,spike)
        # potential,spike=self.conv2(potential,spike,is_training=is_training)
        # #self.see_potential_evolve(potential)
        # potential,spike=self.pool2(potential,spike)

        return potential,spike

    def feature_extractor(self,image_tensor):
        potential,spike=self.forward(image_tensor)
        feature=spike.sum(dim=0).flatten().to(self.device)
        return feature

    def fit_svm(self,train_data,train_label):
        print(f">>> training SVM (num_samples: {len(train_label)})...")
        if isinstance(train_data, torch.Tensor):
            train_data = train_data.cpu().numpy()
        if isinstance(train_label, torch.Tensor):
            train_label = train_label.cpu().numpy()
        self.svm_classifier.fit(train_data,train_label)
        self.is_svm_trained=True

    def predict(self, image_tensor):
        feature = self.feature_extractor(image_tensor)
        if isinstance(feature, torch.Tensor):
            feature = feature.cpu().numpy()
        return self.svm_classifier.predict(feature.reshape(1,-1))[0]

    def feature_extractor_vdsp(self, image_tensor):
        potential, spike = self.forward(image_tensor)
        if spike.dim() == 5:
            spike = spike.squeeze(1)
        return spike.bool().cpu()


    def see_potential_evolve(self, potential, num_to_see=5):
        potential_2d = potential.view(self.timesteps, -1)

        potential_2d = potential_2d.transpose(0, 1)

        indices = torch.randperm(potential_2d.shape[0], device=self.device)[:num_to_see]
        fig, axes = plt.subplots(num_to_see, 1, figsize=(12, 12))

        for i in range(num_to_see):
            axes[i].plot(potential_2d[indices[i], :].cpu().numpy())
            axes[i].set_ylim(-2, 12)
            axes[i].set_title(f"Potential of neuron No. {indices[i]}")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('potential_evolve.png')
        plt.show()
        input("Potential image saved. Press Enter to exit...")
        sys.exit()
