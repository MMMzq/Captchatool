from flask import Flask,request
import commod
import os
app=Flask(__name__)
config=commod.read_config_yaml()
path=config['path']['png']


@app.route('/')
def get_index():
    return app.send_static_file('index.html')

@app.route('/init')
def init():
    config=commod.read_config()
    return str(config['index']['png'])

@app.route('/img/<int:i>')
def get_img(i):
    s=request.args['code']
    try:
        os.rename(path+str(i)+'.png',path+s+'.png')
        #已经存在相同的验证码时把该验证码删除
    except FileExistsError:
        os.remove(path+str(i)+'.png')
    return ''

@app.route('/exit/<int:i>')
def exit(i):
    print('save')
    config['index']['png']=i
    commod.save_config_yaml(config)
    return ''

if __name__=='__main__':
    app.run()
    print(path)


