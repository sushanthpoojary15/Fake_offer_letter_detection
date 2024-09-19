from flask import Flask, redirect, render_template,request,url_for
app=Flask(__name__)
friends=["Shwetha","ramith","Sharavanth"]
@app.route("/")
def home():
    name="sushanth"
    gender="male"
    
    return render_template("home.html",name=name,gender=gender,frnd=friends)

@app.route('/submit',methods=['POST'])
def submit():
    name=request.form.get('name')
    friends.append(name)
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
