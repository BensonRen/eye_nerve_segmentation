"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    #linear_unit_list = [ 50, 100, 200, 500]
    #linear_unit_list = [1000, 500]
    #linear_unit_list = [30, 50, 100]
    #reg_scale_list = [5e-4, 1e-3]
    for batch_size in [3]:
        for backbone in ['resnet_18']:
            for bce_weight in [0.5]:
                for pre_train in [True]:
                    flags = flag_reader.read_flag()  	#setting the base case
                    flags.batch_size = batch_size
                    flags.network_backbone = backbone
                    flags.bce_weight = bce_weight
                    flags.pre_train = pre_train
                    flags.model_name = backbone + '_pre_trained_' + str(pre_train) + '_batch_size_' + str(batch_size) + '_bce_weight_' + str(bce_weight) + '_Epoch_' + str(flags.train_step)
                   train.training_from_flag(flags)
                        
    #    for layer_num in range(3,5):
    #for kernel_first in conv_kernel_size_first_list:
    #    for kernel_second in conv_kernel_size_second_list:
    
            
            ############
            # FC layer #
            ############
            #linear = [linear_unit for j in range(layer_num)]        #Set the linear units
            #linear[0] = 8                   # The start of linear
            #linear[-1] = 1                # The end of linear
            #flags.linear = linear

            #####################
            # Convolution layer #
            #####################
            #flags.conv_kernel_size[0] = kernel_first
            #flags.conv_kernel_size[1] = kernel_second
            #flags.conv_kernel_size[2] = kernel_second

            #######
            # Reg #
            #######
            #for reg_scale in reg_scale_list:
            #    flags.reg_scale = reg_scale
            
            ############################
            # Lorentz ratio and weight #
            ############################
            #flags.lor_ratio = ratio
            #flags.lor_weight = weight
            #for j in range(3):
                #flags.model_name ="reg"+ str(flags.reg_scale) + "trail_"+str(j) + "linear_num" + str(layer_num) + "_unit_layer" + str(linear_unit)
            #    flags.model_name ="Lor_ratio_weight_swipe_trail_"+str(j) + "ratio" + str(ratio)[:4] + "weight" + str(weight)
                        
                        
                #flags.model_name ="reg"+ str(flags.reg_scale) + "trail_"+str(j) + "_conv_kernel_swipe[" + str(kernel_first)+ "," + str(kernel_second) + "," + str(kernel_second) + "]"

            #    train.training_from_flag(flags)
