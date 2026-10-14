This is where I'm going to write down my tips/thoughts on the repo as I distill it down to pass to Tom. 

# High Level
The two main idiosyncratic high level decisions for the repo are using pixi and snakemake. 

## Pixi
Pixi is essentially a conda replacement. Now instead of defining the dependencies in an environment, you set them in your repo root through the pixi.toml file. It's super convenient, and helps with reproducibility. To run python scripts, you would run something like "pixi run python src/tm_ml/ingest.py". Here we actually treat everything as modules, (which is why we have the python path line in pixi.toml). Instead of typing that out, you can define a task (like ingest) and type "pixi run ingest" instead. Another element of my pixi usage is that the ml side needs to be split into it's own environment. Essentially this is so pixi correctly picks the gpu (cuda) build of pytorch when it is resolving the dependencies. I tried a couple workarounds to this at one point in the previous repo, and this was all that worked. 

## Snakemake
The main tool for actually running code (ie training models, visualizing their results) is snakemake. Snakemake is a "workflow language". The idea is you define your complicated workflow as different "rules" in your snakefile, and then you can do the whole complicated thing in one step (while still having separate scripts and whatnot). It also generates a graph of what the workflow looks like, you can see the example one in docs/notes.

To use snakemake, you pass in a config file (which would go in config/sweeps), I left gamma.yaml as an example config file that generated the example training runs in results, as well as the resources that you will let it use. The main command you would actually use is something like "pixi run -e ml pipeline --configfile config/sweeps/gamma.yaml -c8 --resources gpu=1". Pipeline is a pixi task you can checkout in pixi.toml, -e ml is telling pixi to use the ml environment. Essentially this will run rule all in the snakefile, which runs the whole training and post-processing workflow in one go.

If you want to really understand snakemake, there are resources online, including a tutorial. I did the tutorial, which was definitely helpful. At this point claude is the one actually writing the snakefile, so how in depth you want to know it is up to you.

### Scheduler Aside
The bio group I'm in has their own cluster with no scheduler, which is mainly what I use, so right now the codebase is not actually set up to run with a scheduler (like SLURM), but it can be. Snakemake and SLURM can go hand in hand, and in fact I set this up on the hotspot side in like the three days before we made the transition matrix pivot (because I was going to move to the universities cluster for the next phase). There are also multiple ways to layer the SLURM stuff vs snakemake, that's something you can figure out between you and claude, hahaha.

# The Workflow
Ok, so say you run a little pixi run -e ml pipeline, what actually happens? First, ingest runs on the raw data if it hasn't already. This mainly splits the data you gave me into matrices and targets. (With snakemake, if the output a rule would make is already there, it doesn't run the rule, so ingest only gets run once). Then, it goes through the config file and finds all the training runs that will need to be done. Each of these will be put into a different directory, you can see the example of this in results/gamma_sweep. From there, it trains each of the runs (or maybe more accurately trains a model with the given hyperparameters of that run). When the training run is done, we run evaluate. Evaluate gets statistics about the run. These are what go in metrics.json, and are things like the test values of different parts of our loss function, or how information is distributed among the latent space dimensions. It's pretty bloated rn too, you can definitely cut it down if you want. We also run the benchmarks rule, which records this information and other info about the training run in docs/benchmarks/runs.csv. I have this mainly for recording how long/how many resources each run takes, so we can make estimates for future runs. Finally we run visualize, which now has become quite large (although I'm handing you a smaller version). I'm realizing maybe I should split it into multiple scripts/rules, but currently it creates all the figures in a run folder. I only tracked these for the first folder in the examples because they take up a fair amount of space, but it creates them for every folder.

# Things I got rid of (in case you want them)
This just lists bells and or whistles I got rid of that you may want. You can find them in the older versions of main.

## Architecture Diagrams
There was an architecture.py in src (and a corresponding snakemake rule) that created an architecture diagram for each run. They're all pretty much the same though, so you can just take one from the slides, or just make a new generic one. 

## Spatial Version From T
I got rid of the functionality that turns T into a spatial version with nodes. It's not really necessary for the machine learning stuff, more just a way of visualizing things.



# Closing Thoughts
That's pretty much what I think you need to know for using the repo. Sorry for the slop, especially in some of the comments claude leaves, if you're going to read through specific code deeply. I am of the opinion that using LLM's for coding is the future (and the present really, hahaha), so essentially all the actual code is written by claude. I am very happy to talk about any questions you have, it's even beneficial to me to answer them as it helps me understand the code base as well. For high level questions, I will be the best source to go to, for a question like what does this specific function in train.py do, claude is probably better. But again, feel free to swing anything by me. I assume we'll also be hopping on a call when I pass this off, but feel free to set one up anytime. 