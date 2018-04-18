# coding: utf-8

import requests
from bs4 import BeautifulSoup
from pandas import DataFrame

#UID:29->tt
url = "http://tt-lab.ist.hokudai.ac.jp/cybozu/cbdb/ag.exe?page=ScheduleUserMonth&UID=29&GID=23&Date=da.2017.5.21&BDate=da.2017.5.21&CP=sgg&SP="

payload = {
#学生のID
"_ID":"87",
"Password":"student",
"_System":"login",
"LoginMethod":"0",
"Submit":"ログイン",
"_Login":"1",
"csrf_ticket":""
}


s = requests.Session()
r = s.post(url=url, data=payload)

soup = BeautifulSoup(r.text, "lxml")
schedule = soup.findAll(class_="eventInner")

title = []
date = []

for tmp in schedule:
    try:
        content = tmp.find(class_="event")
        title.append(content.get("title"))
        date.append("/".join(content.get("href").split("&")[3].split(".")[1:]))
    except:
        pass

out = DataFrame([date, title]).T
out.columns = ["date", "title"]

print(out)