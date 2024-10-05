from fastapi import FastAPI
import pickle
import json


app = FastAPI()


@app.get('/')
def hello_world():
    return "Hello,World"


model = pickle.load(open('prediction.pxl','rb'))

@app.post('/predict')
def insurance_prediction(input_params):
    input_data = input_params.json()
    input_dict = json.loads(input_data)

    age = input_dict['age']
    sex = input_dict['sex']
    bmi = input_dict['weight']/input_dict['height']
    children = input_dict['children']
    smoker = input_dict['smoker']
    region = input_dict['region']

    input_list = [age,sex,bmi,children,smoker,region]

    prediction_result = model.predict([input_list])

    return prediction_result




if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)
