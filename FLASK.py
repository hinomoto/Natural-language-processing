from flask import Flask, render_template #追加
import pymysql #追加

app = Flask(__name__)

@app.route('/')
def hello():

    #db setting
    db = pymysql.connect(
            host='localhost',
            user='root',
            password='●●●',
            db='market_data',
            charset='utf8',
            cursorclass=pymysql.cursors.DictCursor,
        )

    cur = db.cursor()
    sql = "select * from SWAPRATE"
    cur.execute(sql)
    members = cur.fetchall()
    
    print(members)

    cur.close()
    db.close()

    #return name
    return render_template('swaprate.html', title='flask test', members=members) #変更

## おまじない
if __name__ == "__main__":
    app.run(debug=True)