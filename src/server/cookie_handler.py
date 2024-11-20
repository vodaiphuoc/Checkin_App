from pydantic import BaseModel, Field
from typing import Optional, Literal, Union
from enum import Enum
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from fastapi import Request, Response
from copy import deepcopy
from starlette.exceptions import HTTPException
from uuid import uuid4, UUID
from src.firebase import Firebase_Handler
from src.mongodb import Mongo_Handler

class SameSiteEnum(str, Enum):
    lax = "lax"
    strict = "strict"
    none = "none"


class CookieParameters(BaseModel):
    max_age: int = 7 * 24 * 60 * 60  # 14 days in seconds
    path: str = "/"
    domain: Optional[str] = None
    secure: bool = False
    httponly: bool = True
    samesite: SameSiteEnum = SameSiteEnum.lax

class SessionCookie(object):
    def __init__(
        self,
        cookie_name: str,
        secret_key: str,
        db_handler: Union[Firebase_Handler, Mongo_Handler],
        cookie_params= CookieParameters()
        ):
        self.cookie_name = cookie_name
        self.db_handler = db_handler
        self.signer = URLSafeTimedSerializer(secret_key, salt=cookie_name)
        self.cookie_params = deepcopy(cookie_params)

    @property
    def identifier(self) -> str:
        return self._identifier

    def __call__(self, request: Request):
        # Get the signed session id from the session cookie
            signed_session_id = request.cookies.get(self.cookie_name)
            # print("signed_session_id: ",signed_session_id)
            if signed_session_id is None:
                raise HTTPException(status_code=403, detail="Invalid session")
            else:
                try:
                    session_id = UUID(self.signer.loads(signed_session_id,
                                    max_age=self.cookie_params.max_age,
                                    return_timestamp=False
                                    ))
                    found_actions = self.db_handler.searchCookie(signed_session_id)
                    print(signed_session_id)
                    str_actions = ",".join(found_actions)
                    if "signup" in str_actions:
                        return "Signed but have no checkin"
                    elif "checkin" in str_actions:
                        return "Had done checkin before"
                    elif "login" in str_actions:
                        return "Had done login before"
                    else:
                         raise HTTPException(status_code = 403, 
                                             detail="Session is expired, no action found")

                except (SignatureExpired, BadSignature) as e:
                        raise HTTPException(status_code = 403, 
                                            detail="Session is expired, please login again")

    def make_new_session(self,
                         action: Literal['signup','checkin'],
                         response: Response = None,
                         )->None:
        session_id = uuid4()
        if response is not None:
            response.set_cookie(
                key=self.cookie_name,
                value=str(self.signer.dumps(session_id.hex)),
                **dict(self.cookie_params),
            )
        self.db_handler.insertCookie(session_id= str(self.signer.dumps(session_id.hex)), 
                                     action= action)
        return None
