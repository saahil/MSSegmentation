import os

N_FILTERS = [1, 2, 5, 10, 20, 32]
KERNEL_SIZE = [2, 4, 6, 8]
SUBSAMPLE = [2, 3, 4, 5]

for filter_size in N_FILTERS:
    for kernel in KERNEL_SIZE:
        for sampling in SUBSAMPLE:
            yaml_file = open('yaml/convolutional_net2.yaml', 'r')
            yaml = yaml_file.read()
            
            # change number of filters
            yaml = yaml.replace('output_channels: 32', 'output_channels: '+str(filter_size))

            # change kernel size
            yaml = yaml.replace('kernel_shape: [4, 4]', 'kernel_shape: ['+str(kernel)+', '+str(kernel)+']')
            
            # change subsampling size
            yaml = yaml.replace('pool_shape: [2, 2]', 'pool_shape: ['+str(sampling)+', '+str(sampling)+']')
            yaml = yaml.replace('pool_stride: [2, 2]', 'pool_stride: ['+str(sampling)+', '+str(sampling)+']')

            # save as a different model
            yaml = yaml.replace('save_path: "model/bestmodels/CNN-2-55x55x10_clinic.pkl"', 'save_path: "model/bestmodels/CNN-2-55x55x10_'+str(filter_size)+str(kernel)+str(sampling)+'.pkl"')

            yaml_file = open('yaml/convolutional_net2_'+str(filter_size)+str(kernel)+str(sampling)+'.yaml', 'w+')
            yaml_file.write(yaml)
            yaml_file.close()

            os.system('python train.py yaml/convolutional_net2_'+str(filter_size)+str(kernel)+str(sampling)+'.yaml')

