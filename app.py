from flask import Flask, request, send_from_directory
# set the project root directory as the static folder, you can set others.
app = Flask(__name__)

@app.route('/js/<path:path>')
def send_js(path):
  return send_from_directory('js', path)

# @app.route("/dag_view/<path:path>")
# def dag_view(path):
#   s = open('/Users/user/Projects/DataScienceUtils/david/visualize_data/dag.html').read()
#   s = s.replace('PATH', path)
#   return s

@app.route("/tree_view/<path:path>")
def tree_view(path):
  s = open('/Users/daliasmirnov/PycharmProjects/imagenet/tree.html').read()
  s = s.replace('PATH', path)
  return s

@app.route("/thr_view/<path:path>")
def view(path):
  s = open('/Users/daliasmirnov/PycharmProjects/imagenet/img_txt.html').read()
  s = s.replace('PATH', path)
  return s

@app.route('/img/<path:path>')
def send_img(path):
  return send_from_directory('img', path)



if __name__ == '__main__':
  app.run(port=5001)
