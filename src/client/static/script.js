
/*
Define utility functions
image size: width="640" height="480"
*/
//POST request
async function PostSendBlobs(formdata, url, name = null, pass = null, re_pass = null) {
    /*
    Post requests used only for /checkin_process and /login_process
    if target url is /login_process, append name and pass to formdata
    */
    if (url == '/checkin_process') {
        console.log("Run checkin_process branch");
        return await fetch(url, {
            method: 'POST',
            body: formdata,
            });

    } else if (url == '/login_process') {
        console.log("Run login_process branch");
        // formdata.append("user_name", name);
        // formdata.append("user_password", pass);
        return await fetch(url, {
            method: 'POST',
            body: formdata,
            });
    } else {
        console.log("Run signup branch");
        formdata.append("user_name", name);
        formdata.append("user_password", pass);
        formdata.append("user_reinput_password", re_pass);
        return await fetch(url, {
            method: 'POST',
            body: formdata,
            });
    }
};

function testing(url, video_tag_id = "camera-preview", name = null, pass = null) {
    // This function is for testing/draft purpose only
    const vid = document.getElementById(video_tag_id);
    // get stream video from camera
    window.navigator.mediaDevices.getUserMedia({video: true})
        .then((stream) => {
            console.log("get stream");
            vid.srcObject = stream;
            vid.classList.remove('hidden');
        });
};

/*
Main execution
*/
function checkin_face(url, 
                    video_tag_id = 'camera-preview',
                    name = null, 
                    pass = null, 
                    re_pass = null
                ) {
    const track_constraints = {
        'frameRate': 30
    }
    const N = 5;
    // Initialize the camera
    const vid = document.getElementById(video_tag_id);
    // get stream video from camera
    window.navigator.mediaDevices.getUserMedia({video: true})
        .then((stream) => {
            console.log("get stream");
            console.log(name);
            console.log(pass);
            vid.srcObject = stream;
            vid.classList.remove('hidden');
            
            //MediaStream -> MediaStreamTrack
            const track = stream.getVideoTracks()[0];
            // apply constraints to reduce sampleRate
            const blob_promise = track.applyConstraints(track_constraints);
            
            blob_promise.then(() => {
                imageCapture = new ImageCapture(track); // track = list of frame
                return imageCapture;
            })
            .then((imageCapture)=>{
                let body_form = new FormData();
                let internal_counter = 0;
                
                counter = setInterval(get_single_image,400,imageCapture)
                
                // function inside function
                function get_single_image(inputImageCapture) {
                    if (internal_counter == N) {
                        clearInterval(counter);
                        console.log("Clear Counter");
                        const login_response = PostSendBlobs(body_form,url, name, pass, re_pass);
                        login_response.then((value)=>{
                            if (value.ok) {
                                // clear video + turn off camera
                                stream.getTracks()[0].stop();
                                vid.pause();
                                vid.src = null;
                                vid.style.display = "none";
                                // window.document.close();
                                
                                console.log(value);
                                value.text().then((htlm_response)=>{
                                    document.close();
                                    document.write(htlm_response);
                                    document.close();
                                });

                                // Response.redirect("/", 200);
                                // window.location.replace("/");
                                // var status = window.document.getElementById('login-status');
                                // status.value = "Logged OKE";
                            
                            } else {
                                throw new Error('Login failed.');
                            }
                        });
                        

                    } else {
                        internal_counter ++;
                        inputImageCapture.takePhoto()
                        .then(function(blob) {
                            body_form.append("image_blobs",blob);
                        });
                    }
                }
            })
            .catch((error)=>{
                console.error('Failed in get ImageCapture', error);
            });
            
        })
        .catch(error => {
            console.error('Failed to access the camera:', error);
    });
    };