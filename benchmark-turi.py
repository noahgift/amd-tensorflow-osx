import turicreate as tc
import time

def load_data(workers = 32):
    ### Load Data
    start = time.time()
    print(f"Loading images with #{workers}")
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', workers)
    data = tc.image_analysis.load_images('kagglecatsanddogs_3367a/PetImages', 
        with_path=True)
    data['label'] = data['path'].apply(lambda path: 'dog' if '/Dog' in path else 'cat')
    data.save('cats-dogs.sframe')
    print(f"Completed in {time.time()-start}")

def train_gpus():
    ####disable GPU:  Note, Massively slower
    #print("Disabling GPU")
    #tc.config.set_num_gpus(0)
    start = time.time()
    print("Starting Training")
    # Load the data
    data =  tc.SFrame('cats-dogs.sframe')
    # Make a train-test split
    train_data, test_data = data.random_split(0.8)
    # Create the model
    model = tc.image_classifier.create(train_data, target='label')
    print(f"Completed in {time.time()-start}")
train_gpus()