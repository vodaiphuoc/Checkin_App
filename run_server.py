import uvicorn
import ngrok
import asyncio

from src.utils import get_program_config

master_config = get_program_config()
NGROK_AUTH_TOKEN = master_config["ngrok_auth_token"]
HTTPS_SERVER = master_config["https_server"]
APPLICATION_PORT = int(master_config["port"])
DEPLOY_DOMAIN = master_config["deploy_domain"]


async def main():
    config = uvicorn.Config("src.server.server_api:app",
                            host=HTTPS_SERVER,
                            port=APPLICATION_PORT,
                            reload=True,
                            log_level="info",
                            reload_dirs= ["src/client/","src/server"]
                            )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
