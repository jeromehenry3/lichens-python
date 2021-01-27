activate_this = '/var/www/lichens-backend/venv/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))
import logging
import sys
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, '/var/www/lichens-backend/')
sys.path.insert(0, '/var/www/lichens-backend/lichens_api')

#from lichens_api import app as application
import lichens_api
application = lichens_api.app
#application.secret_key = 'anything you wish'

#from flask import Flask
#from flask_cors import CORS
#import lichens
