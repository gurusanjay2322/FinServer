import google.generativeai as genai
genai.configure(api_key="AIzaSyDnocuEMTbVguWoxBwBGICbjk7hxrx7T_c")

for m in genai.list_models():
    print(m.name)
