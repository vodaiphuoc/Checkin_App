from contextlib import asynccontextmanager
import ngrok
from fastapi import FastAPI, Request, Response,Form, File, UploadFile, Body, status, Depends
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from PIL import Image
import io
import numpy as np
from typing import List, Any, Annotated, Literal
import re

from fastapi.middleware.cors import CORSMiddleware
from src.server.cookie_handler import SessionCookie
from starlette.exceptions import HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.utils import get_program_config
from loguru import logger

import glob


master_config = get_program_config()
NGROK_AUTH_TOKEN = master_config["ngrok_auth_token"]
HTTPS_SERVER = master_config["https_server"]
APPLICATION_PORT = int(master_config["port"])
DEPLOY_DOMAIN = master_config["deploy_domain"]

origins = [
    "http://mullet-immortal-labrador.ngrok-free.app/",
    "https://mullet-immortal-labrador.ngrok-free.app/",
    "http://localhost",
    "http://localhost:8080/",
    "http://localhost:8000/",
    "http://127.0.0.1:8000/",
    "http://127.0.0.1:8000"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    proto: "http", "tcp", "tls", "labeled"
    """
    logger.info("Setting up Ngrok Tunnel")
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    ngrok.forward(addr = HTTPS_SERVER+':'+str(APPLICATION_PORT),
                  proto = "http",
                  domain = DEPLOY_DOMAIN
                  )
    # init engine here
    from src.server.checkin import Checking_Engine
    from src.mongodb import Mongo_Handler

    app.sub_http_pattern = r"|".join([r"\b{}\b".format(ele) for ele in ['http://','https://']])
    app.origin_re_pattern = r"|".join([r"\b{}\b".format(re.sub(app.sub_http_pattern,"",ele).replace('/','')) 
                                        for ele in origins])

    app.db_handler = Mongo_Handler(master_config= master_config, 
                                   ini_push= False)
    
    app.checkin_engine = Checking_Engine(master_config= master_config, 
                                        db_handler = app.db_handler,
                                        running_mode= 'checkin'
                                        )
    
    app.cookie_ssesion = SessionCookie(cookie_name = "app_cookie",
                                       secret_key = "DONOTUSE",
                                       db_handler = app.db_handler)

    yield
    logger.info("Tearing Down Ngrok Tunnel")
    app.db_handler.close()
    app.checkin_engine = None
    app.cookie_ssesion = None
    ngrok.disconnect()


app = FastAPI(lifespan=lifespan)
app.mount(path = '/templates', 
          app = StaticFiles(directory='src/client/templates', html = True), 
          name='templates')

app.mount(path = '/static', 
          app = StaticFiles(directory='src/client/static'),
          name='static')

app.mount(path = '/client', 
          app = StaticFiles(directory='src/client'),
          name='client')

app.add_middleware(CORSMiddleware, 
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )

templates = Jinja2Templates(directory='src/client/templates')


@app.post("/signup")
async def sign_up(request: Request,
                response: Response,
                user_name: Annotated[str, Form()],
                user_password: Annotated[str, Form()],
                reinput_password: Annotated[str, Form()],
                image_blobs: Annotated[list[UploadFile], Form()]
                ):
    if user_password != reinput_password:
        return templates.TemplateResponse(request=request,
                                    name="signup_page.html",
                                    context= {'Login_status': f'The password does not match.\
                                              Please signup again'})

    elif request.app.db_handler.check_duplicate_name_password(user_name, user_password):
        # has duplicates
        return templates.TemplateResponse(request=request,
                                    name="signup_page.html",
                                    context= {'Login_status': f'Your name input exist in database.\
                                            Please signup again'})

    else:
        batch_images = []
        for blob in image_blobs:
            image_content = await blob.read()
            img = np.array(Image.open(io.BytesIO(image_content)))
            batch_images.append(img)
        
        embeddings = request.app.checkin_engine(input_images= batch_images,
                                                return_embeddings_only=True)
        # embeddings = None
        # no duplicate branch
        request.app.db_handler.insertInforMany(user_name= user_name,
                                                images= batch_images,
                                                embeddings= embeddings,
                                                password= user_password
        )
        
        request.app.cookie_ssesion.make_new_session(response= response, action= 'signup')
        
        return RedirectResponse(url=app.url_path_for('homepage_router'),
                                headers= response.headers,
                                status_code=status.HTTP_303_SEE_OTHER)

# in case of user use app on other devices (already has an account) 
# or just the first time use, there will have exception 'invalid session'.
# Thus need to re-render index.html with two option: signup and login
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """This exception handler is use for globaly"""
    msg_detail = str(exc.detail)
    logger.info("In exception handler: ",msg_detail)
    if msg_detail == "Session is expired, please login again" or \
        msg_detail == "Session is expired, no action found":
        logger.info("Exception at this branch")
        return templates.TemplateResponse(request=request,
                                        name="login_page.html",
                                        context= {'Login_status': msg_detail})

    elif msg_detail == "Invalid session":
        # invalid session case
        invalid_case_context = {'Login_status': msg_detail,
            'signup_message' : 'Create a new account if you dont have',
            'keyword': 'login',
            'message': 'Already an account and use on this new device. Please login',
            'BUTTON_NAME': 'LOGIN'
        }
        return templates.TemplateResponse(request=request,
                                        name="index.html",
                                        context= invalid_case_context)
    elif msg_detail == "forbidden":
        return JSONResponse(status_code= exc.status_code, content= "Forbid to see this page")

@app.get(path= "/",
         response_class=HTMLResponse)
async def homepage_router(request: Request):
    
    msg = request.app.cookie_ssesion(request)
    print("home page msg:",msg)

    if msg == "Signed but have no checkin":
        signed_case_context = {'Login_status': 'You has signed up so can do check-in now',
            'signup_message' : '',
            'keyword': 'checkin',
            'message': '',
            'BUTTON_NAME': 'CHECKIN'
        }
        return templates.TemplateResponse(request=request,
                                        name="index.html",
                                        context= signed_case_context)
    else:
        ok_case_context = {'Login_status': 'You can use app now. Check-in again if you wish to',
            'signup_message' : '',
            'keyword': 'checkin',
            'message': '',
            'BUTTON_NAME': 'CHECKIN'
        }
        return templates.TemplateResponse(request = request,
                                        name = "index.html",
                                        context = ok_case_context)


async def check_referer(request: Request):
    """Dependency for checking referer"""
    header_item_list =  request.headers.items()
    find_referer = [element[0] for element in header_item_list 
                    if element[0] == 'referer' and \
                    re.match(pattern = request.app.origin_re_pattern, 
                             string = re.sub(request.app.sub_http_pattern,"",element[1]).replace('/','')
                             ) is not None
                             
    ]
    if len(find_referer) > 0:
        return True
    else:
        return False

@app.get(path= "/{keyword}", response_class = HTMLResponse)
async def switch_router(request: Request, 
                        keyword:str, 
                        has_referer: Annotated[str, Depends(check_referer)]):
    """Router for switch to checkin and signup page
    To prevent user from directly access to this routes,
    check referer in depedencies
    """
    if has_referer:
        return templates.TemplateResponse(request=request,
                                        name= f"{keyword}_page.html")
    else:
        logger.info(request.headers.items())
        raise StarletteHTTPException(status_code= 403, detail= "forbidden")

@app.post(path= "/login_process", response_class= HTMLResponse)
async def login_router(request: Request,
                         response: Response,
                         user_name: Annotated[str, Form()],
                         user_password: Annotated[str, Form()],
                         image_blobs: Annotated[list[UploadFile], Form()]):
    """Router for getting blob images"""
    batch_images = []
    for blob in image_blobs:
        image_content = await blob.read()
        img = np.array(Image.open(io.BytesIO(image_content))) # BGR format
        batch_images.append(img)
    
    logger.info("Done get batch images")
    predict_user_name = request.app.checkin_engine(input_images= batch_images)
    logger.info("predict user name: ", predict_user_name)

    if not app.db_handler.check_duplicate_name_password(user_name, user_password):
        return templates.TemplateResponse(request=request,
                                    name="login_page.html",
                                    context= {'Login_status': f'Your name does not exist in database.\
                                            Please login again'}
                                    )
    elif predict_user_name != user_name:
        return templates.TemplateResponse(request=request,
                            name="login_page.html",
                            context= {'Login_status': f'Seem like your face does not match with \
                                      your name. Please login again'}
                            )
    else:
        # sucess login branch
        request.app.cookie_ssesion.make_new_session(response = response, action= 'login')
        request.app.cookie_ssesion.make_new_session(response = response, action= 'checkin')
        return templates.TemplateResponse(request = request,
                            name="index.html",
                            context= {'Login_status': f'You ({predict_user_name}) can use app now. \
                                        Check-in again if you wish to'}
                            )

@app.post(path= "/checkin_process", response_class= HTMLResponse)
async def checkin_router(request: Request,
                         response: Response,
                         image_blobs: Annotated[list[UploadFile], Form()]):
    """Router for getting blob images"""
    batch_images = []
    for blob in image_blobs:
        image_content = await blob.read()
        img = np.array(Image.open(io.BytesIO(image_content))) # BGR format
        batch_images.append(img)
    
    predict_user_name = request.app.checkin_engine(input_images= batch_images)
    
    request.app.cookie_ssesion.make_new_session(response= response, action= 'checkin')
    return templates.TemplateResponse(request=request,
                        name="index.html",
                        context= {'Login_status': f'You ({predict_user_name}) can use app now. \
                                    Check-in again if you wish to'}
                        )
