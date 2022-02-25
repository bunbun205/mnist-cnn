#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <ensmallen.hpp>

using namespace mlpack;
using namespace mlpack::ann;

using namespace arma;
using namespace std;

using namespace ens;

arma::Row<size_t> getLabels(arma::mat predOut) {

	arma::Row<size_t> predLables(predOut.n_cols);

	for(arma::uword i = 0; i < predOut.n_cols; ++i) {

		predLables(i) = predOut.col(i).index_max() + 1;
	}

	return predLables;
}

int main() {

	constexpr double RATIO = 0.1;
	constexpr int MAX_ITERATIONS = 0;
	constexpr double STEP_SIZE = 1.2e-3;
	constexpr int BATCH_SIZE = 50;

	cout << "Reading data...." << endl;

	mat tempDataset;

	data::Load("data/train.csv", tempDataset, true);

	mat dataset = tempDataset.submat(0, 1, tempDataset.n_rows - 1, tempDataset.n_cols - 1);

	mat train, valid;
	data::Split(dataset, train, valid, RATIO);

	const mat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1);
	const mat validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1);

	const mat trainY = train.row(0) + 1;
	const mat validY = valid.row(0) + 1;

	FFN<NegativeLogLikelihood<>, RandomInitialization> model;

	model.Add<Convolution<>>(1,
				 6,
				 5,
				 5,
				 1,
				 1,
				 0,
				 0,
				 28,
				 28);

	model.Add<LeakyReLU<>>();

	model.Add<MaxPooling<>>(2, 2, 2, 2, true);

	model.Add<Convolution<>>(6,
				 16,
				 5,
				 5,
				 1,
				 1,
				 0,
				 0,
				 12,
				 12);

	model.Add<LeakyReLU<>>();

	model.Add<MaxPooling<>>(2, 2, 2, 2, true);

	model.Add<Linear<>>(16 * 4 * 4, 10);
	model.Add<LogSoftMax<>>();

	cout << "Start training...." << endl;

	ens::Adam optimizer(
		STEP_SIZE,
		BATCH_SIZE,
		0.9,
		0.999,
		1e-8,
		MAX_ITERATIONS,
		1e-8,
		true
	);

	model.Train(trainX,
		    trainY,
		    optimizer,
		    ens::PrintLoss(),
		    ens::ProgressBar(),
		    ens::EarlyStopAtMinLoss(
			    [&](const arma::mat&) {

				    double validationLoss = model.Evaluate(validX, validY);
				    std::cout << "Validation loss: " << validationLoss << std::endl;
				    return validationLoss;
			    }
		    ));

	mat predOut;

	model.Predict(trainX, predOut);
	arma::Row<size_t> predLabels = getLabels(predOut);
	double trainAccuracy = arma::accu(predLabels == trainY) / (double)trainY.n_elem * 100;
	model.Predict(validX, predOut);
	predLabels = getLabels(predOut);
	double validAccuracy = arma::accu(predLabels == validY) / (double)validY.n_elem * 100;

	std::cout << "Accuracy: train = " << trainAccuracy << "%, \t valid = " << validAccuracy << "%" << std::endl;

	mlpack::data::Save("model.bin", "model", model, false);

	std::cout << "Predicting...." << std::endl;

	data::Load("data/test.csv", tempDataset, true);

	mat testX = tempDataset.submat(0, 1, tempDataset.n_rows - 1, tempDataset.n_cols - 1);

	mat testPredOut;

	model.Predict(testX, testPredOut);
	Row<size_t> testPred = getLabels(testPredOut);
	std::cout << "Saving predicted labels to \"result.csv\"..." << std::endl;

	testPred.save("results.csv", arma::csv_ascii);
	std::cout << "Nueral networl model is saved to \"model.bin\"" << std::endl;
	std::cout << "Finished" << std::endl;

}