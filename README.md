# Analog Image Classifier 

### Description
This repo contains a model to evaluate film photographs, inspired by a personal need to automate the management of my film photos. 

### Functionality 
1. Prepare training dataset 
   1. Download film photos from the internet 
   2. Augment photos by adding transformations to overexpose, underexpose and blur training photos
   3. Label photos as `good`, `over_exposed`, `under_exposed`, `light_exposed` or `blurry` 
2. Build model using transfer learning from ResNet50 pretrained model  
3. Train model and evaluate using Weights and Biases 
4. API to interface with model and classify new photos 


### TODO
- [ ] Model sweep using W&B to tune parameters
- [ ] Deploy API 
- [ ] Integrate with web portfolio to automate classification and posting of newly developed film roles  