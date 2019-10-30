# SaliencySamplerProject

To train the model without using the Saliency Sampler:
`python3 inaturalist_no_sampler.py <number of epochs> 0 <batch size> <decimation factor> <state dictionary path (opt)>` 

To train the model with the Saliency Sampler:
`python3 inaturalist_with_sampler.py <number of epochs> <number of epochs with blurring> <batch size> <decimation factor> <state dictionary path (opt)>` 

You  need to download the data and change the path to the data in both training files.
A GPU compatible with the version of PyTorch is neccesary. 

Once the models finish training the results will be in "trained_models" folder. For each run it will produce a results text file and a state dictionary pickle file. 
