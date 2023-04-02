import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='template')
house_data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("My_home_price_model.pickle",'rb'))

@app.route('/')
def index():
    locations = sorted(house_data['location'].unique())
    return render_template('index.html', locations=locations)




try:
    @app.route('/predict', methods=['POST'])
    def predict():
        location = request.form.get('location')
        bhk = float(request.form.get('bhk'))
        bath = float(request.form.get('bath'))
        sqft = request.form.get('total_sqft')

        print(location, bhk, bath, sqft)

        input1 = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input1)
        print(prediction)
        return str(np.round(prediction,2))

except ValueError:
    print("Error")




if __name__ == "__main__":
    app.run(debug=True, port=5001)
