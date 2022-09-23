import numpy as np
import matplotlib.pyplot as plt, mpld3
from tensorflow import keras

import streamlit as st
from streamlit import session_state as session
from mpld3 import plugins
import streamlit.components.v1 as components

# Global vars
current_epoch = 0
x_data = []
y_data = []
math2eng = ''

# Create the Neural Network model 
model = keras.Sequential()
model.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.Dense(units = 1, activation = 'linear'))
model.compile(loss='mse', optimizer="adam")

def next_n_epochs(n, write_model_data=False):
    """ Runs as many of the subsequent epochs as the user desires
    
    Parameters
    ----------
    n : int
        The number of epochs the user wants to run
        The epochs continue from the previous call of this function
    write_model_data : boolean, optional (default is False)
        Whether the user wants to save a text file for each epoch.
        The text file stores model data for that epoch (the weights and biases for each neuron in the NN)
    """
    
    global current_epoch
    the_plot = st.pyplot(plt)

    for x in range(n):
        current_epoch += 1

        # One epoch at a time
        history = model.fit( x_data, y_data, epochs=1, verbose=1)
        loss = round(float(*history.history['loss']),2)

        # Compute the output 
        y_predicted = model.predict(x_data)

        # Display the result
        #fig = plt.figure()
        plt.scatter(x_data[::1], y_data[::1], s=2)
        plt.plot(x_data, y_predicted, 'r', linewidth=4)
        plt.title('{}; Epoch #{}; Loss = {}'.format(math2eng, current_epoch, loss))
        plt.grid()
        the_plot.pyplot(plt)
        plt.clf()

        #time.sleep(0.05)

        filename = f'_1x^2+10cosx-2_5x-E{current_epoch}.txt'

        if (write_model_data):
            with open(filename,'w') as myfile:  
                for layerNum, layer in enumerate(model.layers):
                    weights = layer.get_weights()[0]
                    biases = layer.get_weights()[1]

                    for toNeuronNum, bias in enumerate(biases):
                         myfile.write(f'{layerNum}Bias -> Layer{layerNum+1}Neuron{toNeuronNum}: {bias}')
                         myfile.write('\n')

                    myfile.write('\n')

                    for fromNeuronNum, wgt in enumerate(weights):
                        myfile.write('\n')
                        for toNeuronNum, wgt2 in enumerate(wgt):
                            myfile.write(f'Layer{layerNum}Neuron{fromNeuronNum} -> Layer{layerNum+1}Neuron{toNeuronNum} = {wgt2}')
                            myfile.write('\n')

                    myfile.write('\n')
                    

def watch():
    global x_data, y_data, math2eng

    noise = round(session.noise, 2)
    epochs = session.epochs

    st.text(session.equation)
    st.text(f'Noise = {noise}; Epochs = {epochs}')
    
    # orig is -10,10,num=1000
    x_data = np.linspace(-10, 10, num=1000)
    y_data = (0.1*x_data*x_data) + (10 * np.cos(x_data)) - (2.5*x_data) + noise*np.random.normal(size=1000)
    math2eng = '0.1x^2 * 10cos(x) - 2.5x'

    # Display the dataset
    fig = plt.figure()
    plt.scatter(x_data[::1], y_data[::1], s=2)
    plt.title(f'{math2eng} + {noise} of randomness')
    plt.grid()

    # st.pyplot(fig)

    next_n_epochs(epochs)

    
def user_input():
    with st.form(key='user_input'):
        equation_list = ["0.1x^2 * 10cos(x) - 2.5x", "3* x_data * np.tanh(x_data)"]
        st.selectbox("Select your equation", equation_list, key="equation")
        st.number_input('Choose your noise level (the higher the number the longer the training)', 
            key='noise', min_value=0.1, max_value=5.0, step=0.1)
        st.number_input('Choose the number of epochs (the higher the number the more the time the model gets to learn)', 
            key='epochs', min_value=10, max_value=300, step=10)
        submit_file_btn = st.form_submit_button(label="Let's Watch!", on_click=watch)

if __name__ =="__main__":
    st.set_page_config(layout="wide")
    st.title("Watch a Machine Learn!")
    st.subheader("This interactive application will help you to visualize what it means when people use the terms ML and AI.")

    # Display the model
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    st.text(f'This is the Neural Network we are using: {short_model_summary}')
    st.text("")
    st.text("")

    st.text("So let's get started!  Choose an equation, some random noise and the epochs:")

    user_input()



    ############################



    def f(t):
        return np.exp(-t) * np.cos(2*np.pi*t)

    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)

    two_subplot_fig = plt.figure(figsize=(8,8))
    plt.subplot(211)
    plt.plot(t1, f(t1), color='tab:blue', marker=',')
    plt.plot(t2, f(t2), color='black', marker='.')

    plt.subplot(212)
    plt.plot(t2, np.cos(2*np.pi*t2), color='tab:orange', linestyle='--', marker='.')

    # Define some CSS to control our custom labels
    css = '''
    table
    {
    border-collapse: collapse;
    }
    th
    {
    color: #ffffff;
    background-color: #000000;
    }
    td
    {
    background-color: #cccccc;
    }
    table, th, td
    {
    font-family:Arial, Helvetica, sans-serif;
    border: 1px solid black;
    text-align: right;
    }
    '''

    for axes in two_subplot_fig.axes:
        for line in axes.get_lines():
            xy_data = line.get_xydata()
            labels = []
            for x,y in xy_data:
                html_label = f'<table border="1" class="dataframe"> <thead> <tr style="text-align: right;"> </thead> <tbody> <tr> <th>x</th> <td>{x}</td> </tr> <tr> <th>y</th> <td>{y}</td> </tr> </tbody> </table>'
                labels.append(html_label)
            tooltip = plugins.PointHTMLTooltip(line, labels, css=css)
            plugins.connect(two_subplot_fig, tooltip)

    #fig_html = mpld3.fig_to_html(two_subplot_fig)
    #components.html(fig_html, height=850, width=850)

