**Face recognition model with many detectors and recognition tensorflow model on CPU**

To install dependencies, using pip install -r requirements.txt





# offline
RGB -> cv.imread -> BGR

cv.imwrite: {RGB -> BGR; BGR -> RGB}

# online
Client -> RGB format -> server




1) Firebase DB
- Storage:
    - users's images (path + object)
    - users/s embeddings (path + object)
- Database:
    index, user_name, image_path, embedding_path (NULL)

2) Web server + model (Horeku)
- FastAPI + uvicorn + ngrok


3) Client (PWA)
- tasks:
    - init page:
        - Sign up button (pending)
        - Sign in button:
            - check internet connection
            - open camera
            - stream image
            - send post reuest


# Processes
- All images -> Horeku -> {embeddings + all images} -> Firebase 
- Register mode:
    - Solution 1:
        - Client  -> captured image -> Firebase; 
        - Firebase -> single image -> Horeku -> embedding -> Firebase
    - Solution 2:
        - Client  -> captured image -> Horeku -> {embedding + captured image} -> Firebase
 
- Check-in mode:
    - Init Horeku:
        Firebase -> All embeddings -> Horeku -> embedding layer
    - Client side:
        - Client  -> captured image -> Horeku -> Dot product -> score;


# For facenet model
running in .h5 model or with .tflite model is not much different
since the model is still float32 dtype. Note that post-training
quantization requires a dataset (and a loss function also)


# Deploy link
https://mullet-immortal-labrador.ngrok-free.app/


# References
- code development:
    https://github.com/Rishit-dagli/Face-Recognition_Flutter?tab=readme-ov-file#a-simple-home-page-ui
    https://github.com/timesler/facenet-pytorch

- model devlopement:
    https://github.com/timesler/facenet-pytorch.git

- firebase:
    https://github.com/firebase/firebase-admin-python

- fastapi:
    https://fastapi.tiangolo.com/

- ngrok:
    https://ngrok.github.io/ngrok-python/
    https://dashboard.ngrok.com/get-started/setup/windows

- uvicorn: 
    https://www.uvicorn.org/

- PWA:
    https://blog.heroku.com/how-to-make-progressive-web-app