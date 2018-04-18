# -*- coding: utf-8 -*-

import datetime
import smtplib
import platform
from email.mime.text import MIMEText

from logging import getLogger, DEBUG, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())
logger.setLevel(DEBUG)

def send_calculation_report(text):

    ###########
    # setting #
    ###########

    SENDER_LOGIN_ID ="active mail login ID"
    SENDER_PASSWORD ="active mail login password"
    SENDER_ADDRESS = "my ist mail address"
    RECIEVER_ADDRESSES = "receiver mail address"
    SENDER_DOMAIN = "smtp.hines.hokudai.ac.jp"
    
    time = datetime.datetime.today()
    email_body = """Calculation Report {}-{} {}:{}:{}
    
{}
    
Windows: {}
node: {}
Python: {}
    """.format(time.month, time.day, time.hour, time.minute, time.second, text, platform.uname().version, platform.uname().node, platform.python_version())
    
    msg = MIMEText(email_body, 'plain', 'utf-8')
    logger.debug(email_body)
    msg['Subject'] = "Calculation Report by Python"
    msg['From'] = SENDER_ADDRESS
    msg['To'] = RECIEVER_ADDRESSES

    try:
        smtp_auth_client = smtplib.SMTP(SENDER_DOMAIN, 587)
        smtp_auth_client.ehlo()
        smtp_auth_client.starttls()
        smtp_auth_client.login(SENDER_LOGIN_ID, SENDER_PASSWORD)
        logger.debug("Login succeeded")
        smtp_auth_client.sendmail(SENDER_ADDRESS, RECIEVER_ADDRESSES, msg.as_string())
        logger.debug("Finish sending from {} to {}".format(msg["From"], msg["To"]))
        smtp_auth_client.quit()
        logger.debug("Session terminated")
    except:
        logger.error("Sending failed", exc_info=True)