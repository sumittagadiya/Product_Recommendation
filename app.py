import sqlite3
from flask import Flask, request,url_for, abort, jsonify,render_template
from werkzeug.exceptions import abort
import warnings
warnings.filterwarnings('ignore')
from ml_model import find_similarity




app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('my_data.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/products',methods=['GET'])
def all_products():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM aaic ORDER BY formatted_price DESC LIMIT 20').fetchall()
    conn.close()
    if posts is None:
        abort(404)
    return render_template('all_products.html', posts=posts)

@app.route('/product_info/<int:pid>',methods=['GET'])
def get_selected_product(pid):
    conn = get_db_connection()
    post = conn.execute('SELECT * FROM aaic WHERE id = ?',(pid,)).fetchall()
    # get top_10 most similar item id
    top_10_id = find_similarity(post[0]['title'],10)
    similar_post = conn.execute("select * from aaic where id in (%s)" % (', '.join(str(id) for id in top_10_id))).fetchall()
    conn.close()
    if post is None:
        abort(404)
    return render_template('product_details.html', posts=post,prosim=similar_post)

if __name__ == "__main__":
    app.run(host='localhost',port=8080,debug=True)

