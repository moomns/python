# -*- coding: utf-8 -*-

def show_member_information(obj):
    members = obj.__dict__.keys()
    for key in members:
        print(key, ":\t", eval("self.{}".format(key)))
