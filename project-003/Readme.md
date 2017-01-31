# Driving Simulator Behavioral Cloning

To perform end-to-end autonomous vehicle control in simulation, I employed a NN
starting from NVIDIAs DAVE 2 network architecture. Given that it worked in the
real world, I figured it would at least be a functional starting point for
simulated autonomous driving (which it was).

Here is [a video](https://youtu.be/pEzpXd7A1Zo) of it doing two laps on my 15
inch Macbook Pro.

## Network Architecture
The network architecture is comprised of 5 convolutional layers followed by a
flattening layer and then 3 fully connected layers, finally outputing a single
number for steering angle. Inbetween all layers, I included a relu activation to
support non-linearities.

This architecture managed to get me all the way to a fully functional model that
could perform several laps around track 1 without touching the curb or driving
off the track. I experimented with different numbers and sizes of fully
connected and convolutional layers and did not see much improvement (in most
cases it got worse).

I was able to train this network on an AWS g2.xlarge machine on 130k images at
~70 second/epoch, which meant that I could try out different models and datasets
every 10-15 minutes, which helped me learn more rapidly about what was working
and what was not.

I experimented with adding dropout between the layers, however I found the my
model simply "converged" on a single output answer for all inputs, drastically
underfitting the model to the data and since I was only shooting for the first
track (see also Summary section re: track 2), I did not have many issues with
overfitting.

## Data Sets

I experimented with 4 total data sets for training and validation, while using
the simulator itself for testing. Since the goal was to build a network that
could complete at least one lap, that felt it was a best test, although I would
have much preferred a more automated and repeatable test as that would have let
me compare one network to another in a more quantitative, repeatable, objective,
and auotmated manner.

The datasets were:

1. *"bc_track"*: clean laps of track 1 done by Joshua Owoyemi (@toluwajosh,
thank you Joshua!!), using the center images as is, while adding 0.25 and -0.25
to the steering angles of the left and right images respectively.

2. *"udacity"*: the udacity provided data set, with the same center, left, and
right processing.

3. *"smooth"*: a dataset I generated of 35k images driving a cleanly, smoothly
and as centered on the track as I could using an analog joystick. Processed the
same as set 1 and 2.

4 *"recovery"*: a dataset I generated of 23k images of the car nearly driving
off the track, but recovering before doing so. To ensure I was not teaching the
network to drive towards the edge of the track, I discarded all images with a
zero steering angle and I made sure not only steer the vehicle once it was about
to drive off the track. Discarding the zero-angle images, reduced the total size
of the dataset down to only a few thousand images. For this set, I discarded the
left and right images.

All of the data was preprocessed to crop out the top and bottom portion of the
image to match the 66px height of the network and then resize the 320px width
down to 200px to match the network width. I experimented with RGB and HSV color
spaces and found there to be very litte effect on the performance. Since all of
the data I used could fit into the 15GB of memory on the AWS instance, I did not
need a fit generator.

Finally, each time, all of the training data was shuffled and then 20% was split
off for validation.

## Training

The model using an Adam optimizer and mean square error as its loss function and
tested with a range of 5 to 50 epochs, ultimately settling on 10 as a compromise
between minimizing training loss vs reducing overfitting.

I had mixed success with the datasets. I found that just using "bc_track" alone
was enough to make it around the track, although the car swerved a lot and
nearly drove onto the curb on the tightest right hand corner of the track. As a
result, I decided to add in the other datasets and with that, the model was
smoother, but did in fact drive off during tight right hander.

As a result, I reverted back to just using set 1 for the final model. Given more
time, faster training, and a more repeatable test procedure, I definitely think
I could generate more/different data and improve the model. Alas, I stopped
tinkering and testing at a certain point and called it "done".

## Summary
I found this to be a challenging adn engaging project. I started off nearly
feeling like I would not be able to complete the project, but finished feeling
like I had a *MUCH* stronger grasp on the concepts of NNs, training data,
behavioral cloning, and Keras as a framework.

I did notice (like several other students) that my models performed better if
the simulator was running in lower quality+resolution, perhaps because the CPU
was more available and therefore returned predictions to the simulator with
lower latency.

Some things I would have liked to do, but decided to skip after ~20 hours on the
project:

* *Simpler/smaller networks*: I suspect this might make training much faster and
 also might make the network make predictions faster

* *More repeatable tests*: I would certainly work to create a set of test data
 and run all models against it as a comparison. I would even like to try a
 meta-optimization of automatically running and tuning hyper parameters to
 minimize loss on the test set.

* *Track 2*: I tried this track a bit, but found my model could not complete it.
 I think with more work to reduce over-fitting and possibly some improvements to
 datasets and hyper parameters, the model could complete track 2.

* *Psuedo-random initialization*: I found that training was
 highly non-repeatable because the initialization of the model parameters was
 random. As a result, training the same model on the same data more than once
 resulted in very different performance in the simulator. To make this process
 more predictable (and therefore more in the engineers control), I would like to
 have experimented with making these inputs psuedo random and making their
 random seed a hyper paramenter (which could be meta-optimized, like I described
 above).

