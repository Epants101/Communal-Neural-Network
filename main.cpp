/****************************************************************************************************************/
/*                                                                                                              */ 
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   B L A N K   A P P L I C A T I O N                                                                          */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */  
/****************************************************************************************************************/

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>

#include <stdint.h>
#include <limits.h>

// OpenNN includes
//deduces and nor xor which equal 1

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;

int main(void)
{

    try
    {
        std::cout << "OpenNN. Blank Application." << std::endl;

		srand((unsigned int)time(NULL));

		DataSet data;
		NeuralNetwork network(2, 5, 3);

		//replace hardcode with the location of your data
		data.set_data_file_name("data\\andnorxor.csv");
		data.set_separator(DataSet::Comma);
		data.load_data();

		//label variables
		Variables* variables_pointer = data.get_variables_pointer();

		variables_pointer->set_name(0, "a");
		variables_pointer->set_use(0, Variables::Input);

		variables_pointer->set_name(1, "b");
		variables_pointer->set_use(1, Variables::Input);

		variables_pointer->set_name(2, "and");
		variables_pointer->set_use(2, Variables::Target);

		variables_pointer->set_name(3, "nor");
		variables_pointer->set_use(3, Variables::Target);

		variables_pointer->set_name(4, "xor");
		variables_pointer->set_use(4, Variables::Target);

		//setup the logistics function
		network.get_multilayer_perceptron_pointer()->get_layer_pointer(0)->set_activation_function(Perceptron::ActivationFunction::Logistic);
		network.get_multilayer_perceptron_pointer()->get_layer_pointer(1)->set_activation_function(Perceptron::ActivationFunction::Logistic);

		//Connect up the inputs and outputs
		Inputs* inputs_pointer = network.get_inputs_pointer();

		inputs_pointer->set_information(variables_pointer->arrange_inputs_information());

		Outputs* outputs_pointer = network.get_outputs_pointer();

		outputs_pointer->set_information(variables_pointer->arrange_targets_information());

		//Setup the scaling and probabilistic layers
		network.construct_scaling_layer();

		ScalingLayer* scaling_layer_pointer = network.get_scaling_layer_pointer();

		scaling_layer_pointer->set_statistics(data.scale_inputs_minimum_maximum());

		scaling_layer_pointer->set_scaling_method(ScalingLayer::NoScaling);

		network.construct_probabilistic_layer();

		ProbabilisticLayer* probabilistic_layer_pointer = network.get_probabilistic_layer_pointer();

		probabilistic_layer_pointer->set_probabilistic_method(ProbabilisticLayer::Softmax);

		network.save("data\\neural_network_init.xml");

		//loss index
		LossIndex lossIndex = LossIndex(&network, &data);

		//training strategy
		TrainingStrategy trainer = TrainingStrategy(&lossIndex);

		trainer.set_main_type(TrainingStrategy::GRADIENT_DESCENT);

		GradientDescent* gradientDescentPointer = trainer.get_gradient_descent_pointer();

		gradientDescentPointer->set_minimum_loss_increase(1.0e-200);
		gradientDescentPointer->set_loss_goal(1.0e-30);
		
		trainer.get_loss_index_pointer()->set_error_type(LossIndex::ErrorType::NORMALIZED_SQUARED_ERROR);

		TrainingStrategy::Results results = trainer.perform_training();

		cout << results.gradient_descent_results_pointer->loss_history << endl;

		//Save results

		data.save("data\\data_set.xml");

		network.save("data\\neural_network.xml");

		trainer.save("data\\training_strategy.xml");

		double firstNum;
		double secondNum;
		Vector<double> inputs;
		Vector<double> outputs;

		while (true) {
			cout << "Write two binary digits followed by a space each and you will get you answer" << endl;
			cin >> firstNum >> secondNum;

			cout << endl << outputs << endl;
		}

        return(0);
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;

        return(1);
    }

}  


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2016 Roberto Lopez.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
