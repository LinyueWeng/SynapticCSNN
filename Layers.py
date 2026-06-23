import os
from matplotlib.lines import Line2D
import math

from config import MONITOR_SYNAPSE

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import sys
from torch.nn.functional import conv2d, max_pool2d, unfold
from Synapse_Models import Ferroelectric, Ferroelectric_Tanh
import matplotlib.pyplot as plt
import imageio
import io
from Characterization import ModelCharac
from Characterization import MODEL_CONFIGS
#device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_local = 'cpu'


class CsnnLayer:
    """
    Represents a layer in a convolutional spiking neural network (CSNN) with specific model characteristics.

    This class simulates a spiking neural network layer, incorporating detailed synapse
    models and weight adjustments. It includes mechanisms for lateral inhibition, weight
    updates, defect handling, and device-specific parameter variations. The layer operates
    on input potentials and input spikes during each forward operation.

    :ivar device: Computational device used for running tensors and operations (e.g., CPU or GPU).
    :type device: torch.device
    :ivar batch_size: Batch size of input data.
    :type batch_size: int
    :ivar input_c: Number of input channels.
    :type input_c: int
    :ivar input_h: Height of the input tensor.
    :type input_h: int
    :ivar input_w: Width of the input tensor.
    :type input_w: int
    :ivar output_channels: Number of output channels produced by this layer.
    :type output_channels: int
    :ivar kernel_size: Size of the convolutional kernel.
    :type kernel_size: int
    :ivar stride: Step size of the convolution kernel during forward pass.
    :type stride: int
    :ivar padding: Padding added to the input tensor before convolution.
    :type padding: int
    :ivar timesteps: Number of time steps for spiking computation.
    :type timesteps: int
    :ivar r_inhib: Radius of local inhibition applied to the layer.
    :type r_inhib: int
    :ivar lr: Learning rate for synapse weight updates.
    :type lr: float
    :ivar n_winners: Number of neurons selected as winners during lateral inhibition.
    :type n_winners: int
    :ivar sfp: Scaling factor parameter specific to the model.
    :type sfp: float
    :ivar v: Voltage-dependent scaling factor.
    :type v: float
    :ivar sfd: Scaling factor parameter for defect variations.
    :type sfd: float
    :ivar tau: Time constant influencing potential decay.
    :type tau: float
    :ivar w_sat: Saturation value for the weights.
    :type w_sat: float
    :ivar v_rest: Resting potential value of the neurons.
    :type v_rest: torch.Tensor
    :ivar v_thresh: Threshold value for spiking potential.
    :type v_thresh: torch.Tensor
    :ivar v_reset: Reset value for the potential after a spike.
    :type v_reset: torch.Tensor
    :ivar w_min: Minimum allowable value for the synapse weights.
    :type w_min: torch.Tensor
    :ivar w_max: Maximum allowable value for the synapse weights.
    :type w_max: torch.Tensor
    :ivar f_dep: Factor for dependency adjustment during operations.
    :type f_dep: torch.Tensor
    :ivar synapse_model: Name of the synapse model used in the layer.
    :type synapse_model: str
    :ivar base_params: Dictionary containing base parameters of the synapse model.
    :type base_params: dict
    :ivar model_config: Configuration dictionary for the selected synapse model.
    :type model_config: dict
    :ivar variations_config: Variation parameters for specific devices in the synapse model.
    :type variations_config: dict
    :ivar device_vc_list: Coefficients for device-to-device variation in the model.
    :type device_vc_list: dict
    :ivar cycle_vc: Coefficient for cycle-to-cycle variation in the model.
    :type cycle_vc: float
    :ivar model_charac: Characteristics of the synapse model with applied variations.
    :type model_charac: dict
    :ivar defect_mask: Mask indicating defect-related neurons.
    :type defect_mask: torch.Tensor or None
    :ivar defect_account: Count of defective neurons in the layer.
    :type defect_account: torch.Tensor
    :ivar defect_ratio: Ratio of defective neurons in the total weight tensor.
    :type defect_ratio: torch.Tensor
    :ivar weight: Weight tensor for the convolution operations.
    :type weight: torch.Tensor
    :ivar output_h: Computed height of the output tensor after convolution.
    :type output_h: int
    :ivar output_w: Computed width of the output tensor after convolution.
    :type output_w: int
    :ivar potential: Tensor representing the potential values of neurons.
    :type potential: torch.Tensor or None
    :ivar activation: Tensor indicating the activation state of neurons.
    :type activation: torch.Tensor or None
    :ivar vdsp_neurons: Mask of neurons involved in voltage-dependent spike processing.
    :type vdsp_neurons: torch.Tensor or None
    :ivar target_channel: Target channel index for specific experiments.
    :type target_channel: int
    :ivar delta_weight_list: List of weight deltas computed during training.
    :type delta_weight_list: list
    :ivar frames: Collection of frames to visualize training or weights.
    :type frames: list
    :ivar update_counter: Counter tracking weight update operations.
    :type update_counter: int
    """
    def __init__(self,
                 input_shape,
                 output_channels,
                 kernel_size=7,
                 stride=1,
                 padding=3,
                 lr=0.01,
                 f_dep=2,
                 timesteps=15,
                 v_rest=0, v_thresh=1,v_reset=-1,
                 r_inhib=3, n_winners=7,
                 w_min=0, w_max=0.1,w_sat=1.0,
                 weight_mean=0.08, weight_std=0.005,
                 sfp=1.038,sfd=1.30,
                 v=1.0,
                 tau=0.99,
                 device=device_local,
                 synapse_model='Softbound'):

        self.monitor_synapse=MONITOR_SYNAPSE
        self.end_monitor=False

        self.device = torch.device(device) if isinstance(device, str) else device
        self.batch_size, self.input_c, self.input_h, self.input_w = input_shape
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.timesteps = timesteps
        self.r_inhib = r_inhib
        self.lr = lr
        self.n_winners = n_winners
        self.sfp = sfp
        self.v = v
        self.sfd = sfd
        self.tau=tau
        self.w_sat=w_sat


        self.v_rest = torch.tensor(v_rest, dtype=torch.float32, device=self.device)
        self.v_thresh = torch.tensor(v_thresh, dtype=torch.float32, device=self.device)
        self.v_reset = torch.tensor(v_reset, dtype=torch.float32, device=self.device)
        self.w_min = torch.tensor(w_min, dtype=torch.float32, device=self.device)
        self.w_max = torch.tensor(w_max, dtype=torch.float32, device=self.device)
        self.f_dep = torch.tensor(f_dep, dtype=torch.float32, device=self.device)

        self.vdsp_cnt = 0
        self.update_lr_cnt = 500
        self.max_lr = 0.1
        self.adaptive_lr = True

        self.synapse_model_dictionary = {
            'Ferroelectric': Ferroelectric,
            'Ferroelectric_Tanh': Ferroelectric_Tanh
        }
        self.synapse_model = synapse_model
        print(f">>> Using Synapse Model: {self.synapse_model}")

        # Model Charac
        charac_model=ModelCharac(self.synapse_model)
        self.base_params=charac_model()
        self.model_config=MODEL_CONFIGS.get(self.synapse_model,{})
        self.variations_config=self.model_config.get('variations', {})
        self.device_vc_list=self.variations_config.get('device_to_device_variation_coefficient',{})
        self.cycle_vc=self.variations_config.get('cycle_to_cycle_variation_coefficient',0.0)
        self.model_charac = self.get_varied_device_params()
        #print(self.model_charac)

        self.weight = torch.normal(mean=weight_mean, std=weight_std,
                                   size=(self.output_channels, self.input_c, self.kernel_size, self.kernel_size)).to(self.device)

        # monitor_initial_conductance=input('Please assign the initial conductance to the monitor synapse')
        # synapse_model_class = self.synapse_model_dictionary.get(self.synapse_model)
        # synapse_instance = synapse_model_class(w_winners, w_sat=self.w_sat, w_max=self.w_max, w_min=self.w_min,
        #                                              v_ref=self.v, sf_p=self.sfp, sf_d=self.sfd, **self.model_charac)
        # self.weight.reshape(-1)[self.monitor_synapse]=synapse_instance.r_to_weff(monitor_initial_conductance)

        """
        Here you may change the criteria to define the defects.
        """
        self.defect_mask=None
        if synapse_model=='Ferroelectric_Tanh':
            self.defect_mask = ((self.model_charac['r_max'] <= self.model_charac['r_min'] ) |
                                (self.model_charac['v0_up']  <= 0) |
                                (self.model_charac['v0_low'] <= 0) |
                                (self.model_charac['voff_up'] >= 0) |
                                (self.model_charac['voff_low'] <= 0))
        elif synapse_model=='Ferroelectric':
            self.defect_mask = ((self.model_charac['gamma_p'] <= 1.0) |
                                (self.model_charac['gamma_d'] <= 1.0) |
                                (self.model_charac['alpha_p']  <= 0) |
                                (self.model_charac['alpha_d'] <= 0) |
                                (self.model_charac['theta_p'] >= 0) |
                                (self.model_charac['theta_d'] <= 0))
        self.defect_account = torch.sum(self.defect_mask)
        self.defect_ratio = self.defect_account / self.weight.numel()
        print("Defect ratio: ", self.defect_ratio.item())
        """
        Here you define how to initialize the weight based on different Models.
        """
        if synapse_model=='Ferroelectric_Tanh':
            ratio=self.base_params['r_max']/(self.base_params['r_min'])/(self.w_max-self.w_min)
            self.weight = torch.clamp(self.weight, min=(1/ratio)/2, max=self.w_sat/2)

        else:
            self.weight = torch.clamp(self.weight, min=self.w_min, max=self.w_max)

        self.output_h = (self.input_h - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.output_w = (self.input_w - self.kernel_size + 2 * self.padding) // self.stride + 1

        self.potential = None
        self.activation = None
        self.vdsp_neurons = None

        self.target_channel = 0
        self.delta_weight_list = []
        self.frames = []
        self.update_counter=0

    def __call__(self, input_potential, input_spike, is_training=False):
        if not isinstance(input_potential, torch.Tensor):
            input_potential = torch.tensor(input_potential, dtype=torch.float32, device=self.device)
        else:
            input_potential = input_potential.to(self.device)

        if not isinstance(input_spike, torch.Tensor):
            input_spike = torch.tensor(input_spike, dtype=torch.float32, device=self.device)
        else:
            input_spike = input_spike.to(self.device)

        self.reset_state()


        if input_spike.dim() == 3:
            input_spike = input_spike.unsqueeze(0)
        if input_potential.dim() == 3:
            input_potential = input_potential.unsqueeze(0)

        timestep, c, h, w = input_spike.shape

        all_potential_updates = conv2d(
            input_spike.float(),
            self.weight,
            stride=self.stride,
            padding=self.padding
        )

        output_spike_list = []
        output_potential_list = []

        for t in range(timestep):
            potential_update = all_potential_updates[t]

            self.potential[self.activation] += potential_update[self.activation]

            spike_mask = self.potential > self.v_thresh
            output_spike = torch.zeros_like(self.potential, dtype=torch.bool, device=self.device)

            if spike_mask.any():
                output_spike = self.lateral_inhibition_forward(spike_mask)

                if is_training:
                    winner_mask = self.get_vdsp_winners_mask()
                    delta_weight = 0
                    if winner_mask.any():
                        delta_weight=self.vsdp_batched(input_potential[t], input_spike[t], winner_mask)
                    """
                    Interpretability research. Unleash the next 2 functions to see the corresponding weight evolution.
                    one function a time only
                    """
                    #self.see_delta_weight_evolve(delta_weight)
                    #self.see_weight_frame()

                self.potential[output_spike] = self.v_reset
                self.activation[output_spike] = False

            output_spike_list.append(output_spike.clone())
            output_potential_list.append(self.potential.clone())

        return torch.stack(output_potential_list), torch.stack(output_spike_list)

    def lateral_inhibition_forward(self, spike_mask):
        """
        Computes the result of the lateral inhibition mechanism during the forward pass.

        This method processes the spike_mask tensor to determine which neurons should
        fire based on their potential values. The neuron with the maximum potential in
        each column is allowed to fire, while other neurons are inhibited. Inhibited
        neurons have their potential reset to the resting value, and their activation
        state is set to False.

        :param spike_mask: A tensor indicating the spike states of neurons. Each element
                           is a boolean-like value representing whether a neuron is
                           spiking (True) or not (False).
        :return: A tensor indicating the inhibited spike states after applying the
                 lateral inhibition mechanism. The shape and structure match the input
                 spike_mask tensor.
        """
        max_pots, max_indices = self.potential.max(dim=0, keepdim=True)
        any_spike = spike_mask.any(dim=0, keepdim=True)

        output_spike = torch.zeros_like(spike_mask, device=self.device)
        output_spike.scatter_(0, max_indices, any_spike)

        inhibited_mask = any_spike & ~output_spike
        self.potential[inhibited_mask] = self.v_rest
        self.activation[inhibited_mask] = False

        return output_spike

    def get_vdsp_winners_mask(self):
        """
        Computes a binary mask indicating the "winner" neurons based on their potentials.

        This function calculates a mask where each True value marks a neuron that has
        been considered as a winner during the iterative selection process. The potential
        values of neurons are evaluated and the neurons with the highest potentials (above
        a given threshold) are marked as winners. The selection is restricted by a radius
        of inhibition, ensuring that no two winners are within a specified distance of
        each other in the output space.

        :returns: A boolean tensor of the same shape as ``self.potential`` where True
                  indicates the neurons selected as winners.
        :rtype: torch.BoolTensor
        """
        winner_mask = torch.zeros_like(self.potential, dtype=torch.bool, device=self.device)
        pots_tmp = self.potential * self.vdsp_neurons

        for _ in range(self.n_winners):
            max_val = pots_tmp.max()
            if max_val <= self.v_thresh:
                break

            flat_idx = torch.argmax(pots_tmp).item()
            c = flat_idx // (self.output_h * self.output_w)
            rem = flat_idx % (self.output_h * self.output_w)
            h = rem // self.output_w
            w = rem % self.output_w

            winner_mask[c, h, w] = True

            h_start = max(0, h - self.r_inhib)
            h_end = min(self.output_h, h + self.r_inhib + 1)
            w_start = max(0, w - self.r_inhib)
            w_end = min(self.output_w, w + self.r_inhib + 1)

            pots_tmp[:, h_start:h_end, w_start:w_end] = 0.0
            pots_tmp[c, :, :] = 0.0

        return winner_mask

    def vsdp_batched(self, input_potential_t, input_spike_t, winner_mask):
        """
        Updates synaptic weights for neurons based on the provided input potential and spike tensors,
        applies synaptic plasticity rules according to the chosen synapse model, and updates internal
        variables such as learning rate and inhibitory neurons.

        :param input_potential_t: torch.Tensor
            Tensor representing the input potentials at the current time step.
        :param input_spike_t: torch.Tensor
            Tensor containing the input spikes at the current time step.
        :param winner_mask: torch.Tensor
            Tensor that marks the winning neurons, typically containing a binary mask.

        :return: torch.Tensor
            Tensor representing the computed weight updates for the winning neurons.
        """
        win_c, win_h, win_w = torch.nonzero(winner_mask, as_tuple=True)
        N = len(win_c)
        if N == 0: return

        self.vdsp_cnt += N
        if self.adaptive_lr:
            old_period = (self.vdsp_cnt - N) // self.update_lr_cnt
            new_period = self.vdsp_cnt // self.update_lr_cnt
            if new_period > old_period:
                self.lr = min(self.lr * (2 ** (new_period - old_period)), self.max_lr)

        lr_t = torch.tensor(self.lr, dtype=torch.float32, device=self.device)

        pad = self.padding
        spk_unfold = unfold(input_spike_t.unsqueeze(0).float(), kernel_size=self.kernel_size, stride=self.stride,
                              padding=pad)
        pot_unfold = unfold(input_potential_t.unsqueeze(0).float(), kernel_size=self.kernel_size, stride=self.stride,
                              padding=pad)

        spatial_indices = (win_h * self.output_w + win_w).long()
        win_spikes = spk_unfold[0, :, spatial_indices].T.view(N, self.input_c, self.kernel_size, self.kernel_size)
        win_pots = pot_unfold[0, :, spatial_indices].T.view(N, self.input_c, self.kernel_size, self.kernel_size)

        w_winners = self.weight[win_c]

        """
        Here you can define the weight update rule based on different synapse models.
        It is convenient to add new models just by adding to the dictionary self.synapse_model_dictionary.
        But you have to keep standardized in&output format of the models.
        """
        if self.synapse_model == 'Softbound':
            cond_pot = win_spikes > 0
            w_factor = w_winners * (self.w_max - w_winners)
            g_dep = self.f_dep - (win_pots / self.v_thresh)

            delta_weights = torch.where(
                cond_pot,
                w_factor * 1.0 * lr_t,
                -w_factor * g_dep * lr_t
            )
        elif synapse_model_class := self.synapse_model_dictionary.get(self.synapse_model):
            cond_pot = win_spikes > 0
            winner_params = {
                k: v[win_c] for k, v in self.model_charac.items()
            }
            synapse_model_instance = synapse_model_class(w_winners,w_sat=self.w_sat,w_max=self.w_max,w_min=self.w_min,v_ref=self.v,sf_p=self.sfp,sf_d=self.sfd,**winner_params)
            delta_weights = synapse_model_instance(win_pots,self.v_reset, self.v_thresh, cond_pot)
        else:
            raise ValueError(f"Synapse model {self.synapse_model} not found")

        self.weight.index_add_(0, win_c, delta_weights)

        self.vdsp_neurons[win_c, :, :] = False

        spatial_mask = torch.zeros((1, 1, self.output_h, self.output_w), dtype=torch.float32, device=self.device)
        spatial_mask[0, 0, win_h, win_w] = 1.0

        if self.r_inhib > 0:
            kernel_size = 2 * self.r_inhib + 1
            dilated_mask = max_pool2d(spatial_mask, kernel_size=kernel_size, stride=1, padding=self.r_inhib)
            dilated_mask = dilated_mask.squeeze(0).squeeze(0).bool()
        else:
            dilated_mask = spatial_mask.squeeze(0).squeeze(0).bool()

        self.vdsp_neurons[:, dilated_mask] = False

        return delta_weights

    def input_spike_generator(self, input_potential):
        input_spike = (input_potential >= self.v_thresh).float()+(input_potential < 0.0).float()
        if input_spike.dim() == 3:
            input_spike = input_spike.unsqueeze(0)
        return input_spike

    def ttfs_inputlayer(self, image_tensors):
        """
        Processes input image tensors to compute input potentials and spike times for a spiking neural network.

        Note: if you set tau=1.0, the input layer could be computed faster than Leaky neurons, this is suitable for quick test and is recommended especially when you do not have a large Timestep and a large target tau value (0.99).

        :param image_tensors: A tensor representing the input images.
                              It can either be a PyTorch tensor or input data in a numeric format
                              that can be converted to a tensor.
        :return: A tuple containing:
                 - input_potential (torch.Tensor): The computed input potentials for the timesteps.
                 - input_spike (torch.Tensor): The spike times generated based on the input potentials.
        """
        if not isinstance(image_tensors, torch.Tensor):
            image_tensors = torch.tensor(image_tensors, dtype=torch.float32, device=self.device)
        else:
            image_tensors = image_tensors.to(self.device)

        if image_tensors.max() > 1.0:
            image_tensors = image_tensors / 255.0

        if image_tensors.dim() == 4:
            image_tensors = image_tensors.squeeze(0)

        image_c, image_h, image_w = image_tensors.shape

        images_spiketime = torch.floor((1.0 - image_tensors) * (self.timesteps - 1))

        steps = torch.arange(1, self.timesteps + 1, dtype=torch.float32, device=self.device).view(-1, 1, 1, 1)

        image_expanded = image_tensors.unsqueeze(0)
        spiketime_expanded = images_spiketime.unsqueeze(0)
        input_potential = torch.zeros(
            self.timesteps, image_c, image_h, image_w,
            dtype=torch.float32, device=self.device
        )
        current = self.v_thresh / (images_spiketime + 1.0)
        if self.tau==1.0:
            input_potential = steps * (self.v_thresh / (spiketime_expanded + 1.0))
            input_potential = torch.clamp(input_potential, max=self.v_thresh.item())

            input_spike = self.input_spike_generator(input_potential)
            input_potential[input_potential==self.v_thresh]=self.v_reset
        else:
            input_potential[0]=(self.v_thresh / (images_spiketime + 1.0))
            for t in range(self.timesteps-1):
                input_potential[t][input_potential[t] >= self.v_thresh] = self.v_reset
                input_potential[t+1][input_potential[t]>=0.0] = current[input_potential[t]>=0.0]+input_potential[t][input_potential[t]>=0.0]*self.tau
                input_potential[t+1][input_potential[t]<0.0] = input_potential[t][input_potential[t]<0.0]*self.tau
            input_spike=self.input_spike_generator(input_potential)
            input_potential[input_potential>=self.v_thresh]=self.v_reset
            input_potential=input_potential.reshape(self.timesteps, image_c, image_h, image_w)
        #print(f"The input potential is {input_potential}")
        return input_potential, input_spike

    def reset_state(self):
        self.potential = torch.ones(self.output_channels, self.output_h, self.output_w, dtype=torch.float32,
                                    device=self.device) * self.v_rest
        self.activation = torch.ones(self.output_channels, self.output_h, self.output_w, dtype=torch.bool,
                                     device=self.device)
        self.vdsp_neurons = torch.ones(self.output_channels, self.output_h, self.output_w, dtype=torch.bool,
                                       device=self.device)

    def get_output_size(self):
        return self.output_channels * self.output_h * self.output_w

    def get_varied_device_params(self):
        """
        Generates varied device parameters adjusted by variations defined in a device-specific variation
        coefficient list and based on base parameter values. This method creates a tensor matrix for each
        parameter with specified shape and variation applied when applicable.

        :raises KeyError:
            If a key in `self.device_vc_list` does not correspond to any entry in `self.base_params`.

        :return: A dictionary of parameters with tensor matrices as values, where each matrix corresponds
                 to a parameter with adjustments applied.
        :rtype: dict
        """
        shape=(self.output_channels,self.input_c,self.kernel_size,self.kernel_size)
        device_params = {}
        for key, value in self.base_params.items():
            variation=abs(self.device_vc_list.get(key,0.0)*value)
            params_matrix = torch.full(shape, value, device=self.device)
            if variation!=0.0:
                params_matrix+=variation*torch.randn(shape,device=self.device) # Here you can switch to other distributions
                #params_matrix = torch.max(params_matrix, torch.tensor(1e-9, device=self.device))
                print(f"The std of device variation of {key} is {variation}")
            device_params[key] = params_matrix
        return device_params



class SnnPooling:
    """
    Max pooling layer for spiking neural networks.

    This class implements a standard pooling layer suited for spiking neural network operations with
    output potential and spike tensors. The outputs are computed in the shape
    `[timesteps, output_channels, output_h, output_w]`. The layer applies max pooling to the inputs,
    tracks spikes, and dynamically adjusts potentials as per spiking thresholds and resets.

    :ivar device: The device upon which computations are performed (e.g., CPU or GPU).
    :type device: Any
    :ivar kernel_size: Size of the pooling kernel.
    :type kernel_size: int
    :ivar stride: Stride value for pooling operation.
    :type stride: int
    :ivar padding: Padding size applied to the input before pooling.
    :type padding: int
    :ivar v_thresh: Threshold potential for spiking.
    :type v_thresh: int
    :ivar v_reset: The value to reset the potential to after a spike.
    :type v_reset: int
    :ivar timesteps: Number of timesteps over which the computations will run.
    :type timesteps: int
    :ivar output_h: Computed height of the output after pooling.
    :type output_h: int
    :ivar output_w: Computed width of the output after pooling.
    :type output_w: int
    :ivar activation: Current activation state of the pooling layer, used to track active neurons.
                     Initialized as None and updated during computations.
    :type activation: Optional[torch.Tensor]
    """
    ## max pooling layer for spiking neural network ##
    ## standard pooling layer with outputs: output_potential and output_spike, all in the shape of [timesteps, output_channels, output_h, output_w]##
    def __init__(self,input_shape,
                 kernel_size=3,
                 stride=1,
                 padding=3,
                 v_thresh=10,
                 v_reset=-1,
                 timesteps=15,
                 device=device_local
                 ):
        self.device=device
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.v_thresh=v_thresh
        self.v_reset=v_reset
        self.timesteps=timesteps

        batch_size,self.input_c,input_h,input_w=input_shape

        self.output_h = (input_h - kernel_size + 2 * padding) // stride + 1
        self.output_w = (input_w - kernel_size + 2 * padding) // stride + 1

        self.activation=None

    def __call__(self,input_potential,input_spike):
        self.reset_state()
        self.activation = torch.ones(self.input_c,self.output_h,self.output_w).bool().to(self.device)

        comp_potential = input_potential.clone()
        comp_potential[input_spike > 0] = self.v_thresh
        pooled_spike = max_pool2d(input_spike.float(), kernel_size=self.kernel_size, stride=self.stride,
                                  padding=self.padding).bool()
        pooled_potential = max_pool2d(comp_potential.float(), kernel_size=self.kernel_size, stride=self.stride,
                                      padding=self.padding)
        cumsum_spikes = torch.cumsum(pooled_spike.int(), dim=0)
        spike_mask = pooled_spike & (cumsum_spikes == 1) & self.activation.unsqueeze(0)
        final_potential = pooled_potential.clone()
        final_potential[spike_mask] = self.v_reset
        has_spiked = cumsum_spikes[-1] > 0
        self.activation = self.activation & (~has_spiked)
        return final_potential.to(self.device), spike_mask.float().to(self.device)

    def reset_state(self):
        self.activation=None

    def get_output_size(self):#if you connect it to readout layer, use this function to get the input size of the pcn layer
        return self.input_c*self.output_h*self.output_w