# 2 D Objecty Detection

Self-driving vehicles need sophisticated perception capabilities to cappture ever-changying dynamics of the surroundings. One task of the perception system - object detection - is the crucial part that allows the vehicle to safelly navigate the environment. 

The project primarily focuses on explopring capabilities of TensorFlow Object Detection API that allows to deploy the model to get the preditions on images sent ot the API. ResNEt50 Neural Networks taken from TensorFlow Model Zoo will be trained to detect three classes (vehicles, pedestrians, cyclists) on a selected subset of Waymo Open dataset.

## Data 

Each tfrecord file contains a mutli-sensor data sequence recorded at a frame of 10 fps. For the training and object detection model only the video data from the front camera including the corresponding meta data and ground truth labels is used. 

The data for training and validation was downsampled only using 1 of every 10 images under the assumption that subsequent images in a video sequence are very similar. 

For training and cross-validation we will use 97 tfrecord files. For testing, we will keep 3 tfrecord files that the model will not see during training.

## Exploratory Data Analysis 

The EDA was conducted in the following notebook:

[Exploratory_Data_Analysis.ipynb](./Exploratory_Data_Analysis.ipynb)

The data set contains urban and motorway scenarios with high to sparse traffic density. The majority of the dataset images were obtained on a clear sunny day. There are, however, some images recorded in less ideal weather conditions that introduce object occlusions. The most prevalent occlusion introduced by the weather conditions was the rain drops visible on the windshield, which resulted in blurred images. Severla snapshots taken at night didn't contain any visually visible vehicles. Headlights created blinding effect at night, which made it hard to distinguish objects. Occasional fog also introduced issues to object detection.

Images also have varying degree of visibility of objects of interest strictly due to their relative position to the camera. Some objects on the pictures are really far away which can make it difficult for the neural network to recognize objects. Some snapshots, on the other hand, didn't contain any oobjects of interest. 

![Brightness distribution](/imgs/brightness_distr.png)
_Upon visual inspection, the average brightness for nighttime condition was determined to be around 70_

Upon looking into the distribution of classes, the dataset turned out to be very unbalanced with vehicles (78.41%) dominating other classes - pedestrains 21% and cyclists 0.59%.

![Labels distribution](/imgs/distribution_of_classes.png)

![Labels chart](/imgs/pie_chart.png)

After conducting the exploratory data analysis, the single most important contrastive aspect in the dataset was determined to be day/nighttime attribute. Hence, both training, validation, and test dataset ideally should contain both scenes record at daytime and nighttime.  

## Subset splits

Ideally, we should end up with approximately similar distribution of all different attributes present in the original dataset in each subset. We would want each split to contain similar distribution of three objects of interest. Additionally, we would want each subset to cover similar distance ranges, weather conditions, bounding boxes positions and sizes. 
The split in the dataset only loosely meets the above described criteria, as the tfrecord files to be used in the test set were already pre-defined. Although they do not contain cyclicts, they represent variety of weather conditions and levels of density of traffic. Even though the random split was utilized to obtain data splits, the features in training and validation sets turned out to be surprisingly equally distributed. 

It should be noted that the above-mentioned approach is by no means an optimal one, but it is sufficient for the goal of this project, which is to explore the capabilities of TensorFlow Object Detection API. The data split allowed us to successsfully experiment which pipeline configurations lead to the overall improvement in performance. 

The experiments described below were conducted on the set of 100 tfrecordf files that were divided into 82 training files, 15 validation files, and 3 test files. Each downsampled tfrecord file contains on average around 16-20 images. The downsampling, even though resulted in optimizing processing time, significantly undermined the overall effectiveness of the system, as the training set and validationb set contained a bit less than 1640 and 300 images correspondingly. The use of augmentations was tested as an  approach to account for the scarcity and unbalanced nature of images presented for training. 

## Data Augmentations

The exploration of the built-in augmentation methods offered by TensorFlow is presented in the following notebook:

[Augmentations.ipynb](./Augmentations.ipynb)

The following data augmentations were chosen upon inspecting results of EDA:
* augmentation methods like random cropping, resizing, flipping, or rotation that try to increase variety of object sizes and object locations on image plane
* augmentation methods that randomly adjust hue, saturation, brightness, or contrast or to randomly distort image colors within some boundaries in order to increase the variety of lighting effects
* augmentation methods like putting a random jitter on the bounding boxes to simulate labeling errors, for example 

Above augmentation methods are tested in the following experiments in order to artificially increase the variety of data samples in our rather very small training and validation data set.

## Transfer Learning Experiments

### Initial Experiment


![Initial Experiment](/imgs/init_tensorboard.png)

Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
Batch size: 4
Optimizer: momentum optimizer 
Scheduler: cosine decay learning rate (learning_rate_base: 0.04, warmup_learning_rate: 0.013333)
Training steps: 2500
Data augmentations default

DetectionBoxes_Precision/mAP - 3.38e-5

DetectionBoxes_Precision/mAP large - 5.37e-4

DetectionBoxes_Precision/mAP medium - 4.89e-5

DetectionBoxes_Precision/mAP small - 0

@.5IOU  1.42e-4

* Classification loss 0.6673
* Localization loss 0.71
* Regularization Loss - 0.7932
* Total Loss: _train_ =  2.189, _eval_ = 2.329

The initial experiment provided baseline results which were not that impressive. The total loss for training and validation set doesn't seem that far off from each other so there is not evident overfitting yet. In was decided to use the PASCAL-VOC metric for future evaluation to gain a better undestanding on how the model performs on three different classes - vehicle, pedestrian, cyclist. The classification by bounding boxes doesn't proviude many insights. 

Since we're repurposing code from other domain it was decided to keep the learning rate constant. The choice of the learning rate highly depends on epoch numbers and size of the dataset. The learning rate was chosen to be smaller (0.01) in an attempt to arrive to a more optimal final set of wrights at the expense of the computational time. In order to achieve a better performmance on the loss function batch size was increased to 8. 

## Experiment 1

Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
Batch size: 8
Optimizer: momentum optimizer 
Learning rate: 0.01
Training steps: 2500
Data augmentations default

![Experiment 1](/imgs/experiment1.png)

Upon using greater batch size, I ran into the GPU memory issue on the Virtual Machine so the training was automatically stopped on 1400 epochs. Even though I was not able to obtain the final metrics value, it is already clear by the loss curves that greater batch size would help to improve the performance of the model. Unfortunately, due to the limitations of the environment, I will have to stick to the batch size of 4. 

## Experiment 2

1. Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
2. Batch size: 4
3. Optimizer: momentum optimizer 
4. Learning rate: 0.01
5. Training steps: 2500
6. Data augmentations default

Using a constant learning rate of 0.01 instead of a cosing decay learning rate lead to a lower loss after the same number of epochs when compared to the initial experiment. It could be infered thyat the base learning rate of the previous setting was too high for the optimization problem. 


![Experiment 2](/imgs/exp2_1.png)
_TensorBoard Visualization_

* Classification loss: train- .3185,    val- .04254
* Localization loss: train- .4309,   val- .5066
* Regularization Loss- train- .3086   val- .3086
* Total Loss - train-1.056    val- 1.239

The loss function looks reasonable but it might have slightly too small of a learning rate based on its speed of decay (even though it might be questionable at this point). It was decided to implement styep decay learning rate annealing. It was modified by dropping the learning rate by a constant whenever the loss function stops imrpoving. The discrepancy between the total loss of training validation set may indicate that the networks starts to overfit. This downside will be taken care of in the later experiments where we will focus on augmentations to fit validation set better.

![Experiment 2](/imgs/exp2_2.png)
![Experiment 2](/imgs/exp2_3.png)

* cyclist: nan
* pedestrian: 5.5e-3
* vehicle: .04418
* Precision: 0.02484

The model shows better perfromance after training the initial pre-trained model but the overall map@-.5IOU is still very low.

Sample detection: 
![Experiment 2](/imgs/exp2_4.png)
![Experiment 2](/imgs/exp2_5.png)

## Experiment 3

1. Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
2. Batch size: 4
3. Optimizer: momentum
4. Learning rate: step-decay 
5. Training steps: 2500
6. Data augmentations default

```
 optimizer {
    momentum_optimizer {
      learning_rate {
        manual_step_learning_rate {
          initial_learning_rate: 0.01
          schedule {
            step: 500
            learning_rate: 0.005
          }
          schedule {
            step: 800
            learning_rate: 0.0025
          }
          schedule {
            step: 1250
            learning_rate: 0.00125
          }
          schedule {
            step: 1600
            learning_rate: 0.000625
          }
        }
      }
      # momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
```

![Experiment 3](/imgs/exp3_1.png)
![Experiment 3](/imgs/exp3_2.png)

Step decay was chosen as a preferred approach to implement annealing learning rate since its parameters are more interpretable and easy to tackle. Exponential and cosine decays, on the other hand, requiure considerable amount of fine-tuning. 

The traininng for that experiment was stopped at around 1.7k steps, as the approached has no indication for the performance improvement. However, if the step decay annealing was set accordinlgy to the optimization problem at hand, it would yield better coinvergence to local minimum. Due to the constraints on the memory resources, other approaches were tested. 

* Classification loss: train- .39    val- .58
* Localization loss: train- .55    val- .6157
* Regularization Loss- train- .7397   val- .741
* Total Loss - train-1.726  val- 1.944

In early stages of setting baselines, Adam optimizer tends to be more forgiving to hyperparameters, including a bad learning rate. For ConvNets, however, a well-tunde SGD will lmost always outperform Adam. Since we were trying to find a better set of hyperparameters, the next experiment explored what effect would Adam optimizier with the initial learning rate of 0.01 have on the overall performance. 

## Experiment 4

1. Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
2. Batch size: 4
3. Optimizer: Adam
4. Learning rate: 0.01 
5. Training steps: 2500
6. Data augmentations default

![Experiment 4](/imgs/exp4.png)

The training was also stopped ahead of the time since the learning curves showed that there will be no considerable improvement in the following number of epochs left. 

When using Adam optimizer instead of momentum optimizer, training loss was slowly decreasing as before but then hit a plateau. Classification loss increased dramatically by the end, which negatively impacted the results. As a conclusion, Adam optimzier was shown to be a poor choice for this optimization problem. 

## Experiment 5

1. Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
2. Batch size: 4
3. Optimizer: momentum
4. Learning rate: cosine decay 
5. Training steps: 2500
6. Data augmentations default

```
optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.01
          total_steps: 2500
          warmup_learning_rate: 0
          warmup_steps: 0
          hold_base_rate_steps: 1100
        }
      }
       momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
```

![Experiment 5](/imgs/exp5_1.png)

![Experiment 5](/imgs/exp5_3.png)
_TensorBoard Training Curves_

* Classification loss: train- .1423    val- .1976
* Localization loss: train- .2042   val- .284
* Regularization Loss- train- .2519   val- .2519
* Total Loss - train- .5983    val-  .7335


![Experiment 5](/imgs/exp5_2.png)

* cyclist: nan
* pedestrian: 0.2
* vehicle: .08446

![Experiment 5](/imgs/exp5_4.png)

Thius experiment yielded the best results yet observed. There is a signifcant decrease in the the total loss and considerable improvement in performance on vehicles and pedestrians. The AP@0.5 IOU on cyclists is still NAN which speaks to the fact that the group was underrepresented in the dataset. 

![Experiment 5](/imgs/exp5_5.png)
![Experiment 5](/imgs/exp5_6.png)

## Experiments with Data Augmentations

Typically, we want to increase the number of steps upon adding augmentations, since the model will be seeing less of the original images. In this case, however, the number of training steps was held constant. 

## Experiment 6

1. Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
2. Batch size: 4
3. Optimizer: momentum
4. Learning rate: cosine decay (as before)
5. Training steps: 2500
6. Data augmentations: see below

```
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_vertical_flip {
    }
  }
  data_augmentation_options {
    random_rotation90 {
    }
  }
  data_augmentation_options {
    random_rotation90 {
    }
  }
  data_augmentation_options {
    random_rotation90 {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
```

![Experiment 6](/imgs/exp6_1.png)

This experiment was focused on expanding the dataset by modifying object size, aspect ratios, and their locations. The experiment hasn't yielded any improvements in results, neither in loss curves nor in the AP&IOU for any of the classes. If trained for greater number of steps, the improvements might have been more noticeable. 

![Experiment 6](/imgs/exp6_3.png)


![Experiment 6](/imgs/exp6_2.png)

## Experiment 7

1. Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
2. Batch size: 4
3. Optimizer: momentum
4. Learning rate: manual step decay (as before)
5. Training steps: 2500
6. Data augmentations: see below

```
 data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  data_augmentation_options {
    random_rgb_to_gray {
    }
  }
  data_augmentation_options {
    random_adjust_brightness {
      max_delta: 0.3
    }
  }
  data_augmentation_options {
    random_adjust_saturation {
      min_delta: 0.7
      max_delta: 1.3
    }
  }
  data_augmentation_options {
    random_adjust_hue {
      max_delta: 0.05
    }
  }
  data_augmentation_options {
    random_adjust_contrast {
      min_delta: 0.5
      max_delta: 1.5
    }
  }
  ```
![Experiment 7](/imgs/exp7_1.png)

![Experiment 7](/imgs/exp7_3.png)
![Experiment 7](/imgs/exp7_2.png)

The data augmentations methods primarily focused on adjusting hue, saturation, brightnest, or contrast. An additional augmentatrions of random image crop and horixontal flip were added. 

From the predictions made after the EDA, it was expected for color-based augmentations to make the biggest improvement on model's performance. Even though the loss curve was still converging, the results after 2500 steps didn't surpass the ones optained in Experiment 5. Since the results were giving hope, the next experiment tried to tweak the learning rate by choosing another annealing method. 

![Experiment 7](/imgs/exp7_4.png)
![Experiment 7](/imgs/exp7_5.png)

## Experiment 8

1. Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
2. Batch size: 4
3. Optimizer: momentum
4. Learning rate: cosine decay (base steps 400)
5. Training steps: 2500
6. Data augmentations: distortion, color augmentations

```
optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.01
          total_steps: 2500
          warmup_learning_rate: 0
          warmup_steps: 0
          hold_base_rate_steps: 400
        }
      }
       momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
```

Augmentations: 

```
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  data_augmentation_options {
    random_rgb_to_gray {
    }
  }
  data_augmentation_options {
    random_distort_color {
      color_ordering: 0
    }
  }
  }
```

![Experiment 8](/imgs/exp8_1.png)

Instead of randomly adjusting colors, the built-in random color distortion was applied, which utilizes automatically predifined ranges. Compared to the previous experiment, the new results point to the fact that automatic distortion could be a bit more effective. 

![Experiment 8](/imgs/exp8_2.png)

## Inference

As the __Experiment 5__ was shown to give the best overall results, an inference video was created to show the model in action. 

![Inference video](/animation3.gif)

The video sequence captures a country road with moderate number of vehicles encountered thorghout the route. The detection works in principle but tends to fail to capture objects that are further away. Although all of the vehicles were identified correctly, the detections' confidence level is quite low. To improve, the dataset should be expanded to include more vehicle ground truth labels (and ideally it should be more balanced). 