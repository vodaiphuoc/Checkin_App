**Face recognition model with many detectors and recognition tensorflow model on CPU**

To install dependencies, using pip install -r requirements.txt


1) Firebase DB
- Storage:
    - users's images (path + object)
    - users/s embeddings (path + object)
- Database:
    index, user_name, image_path, embedding_path (NULL)

2) Web server + model (Horeku)



3) Client (Android App)



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

# References
https://github.com/firebase/firebase-admin-python
https://github.com/Rishit-dagli/Face-Recognition_Flutter?tab=readme-ov-file#a-simple-home-page-ui