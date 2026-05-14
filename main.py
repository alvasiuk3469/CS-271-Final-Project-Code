import copy
import random
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score


def setSeed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


setSeed(42)


@dataclass
class TrainingConfig:
    dataPath: str = "dataset_full.csv"

    batchSize: int = 64
    learningRate: float = 1e-3
    maxEpochs: int = 25
    patience: int = 5

    validationSize: float = 0.15
    testSize: float = 0.15
    randomState: int = 42

    baselineDropout: float = 0.2
    improvedDropout: float = 0.3


config = TrainingConfig()
device = torch.device("cpu")


class Baseline(nn.Module):
    def __init__(self, inputDimension: int, dropout: float = 0.2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(inputDimension, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, inputBatch: torch.Tensor) -> torch.Tensor:
        return self.network(inputBatch)


class Improved(nn.Module):
    def __init__(self, inputDimension: int, dropout: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(inputDimension, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, inputBatch: torch.Tensor) -> torch.Tensor:
        return self.network(inputBatch)


def loadData(dataPath: str) -> Tuple[np.ndarray, np.ndarray]:
    dataFrame = pd.read_csv(dataPath)

    dataFrame = dataFrame.drop_duplicates()

    featureMatrix = dataFrame.drop(columns=["phishing"]).to_numpy(dtype=np.float32)
    targetVector = dataFrame["phishing"].to_numpy(dtype=np.int64)

    return featureMatrix, targetVector


def prepareSplits(featureMatrix: np.ndarray, targetVector: np.ndarray, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    trainValidationFeatures, testFeatures, trainValidationTargets, testTargets = train_test_split(featureMatrix, targetVector, test_size=config.testSize, random_state=config.randomState, stratify=targetVector)

    relativeValidationSize = config.validationSize / (1.0 - config.testSize)

    trainFeatures, validationFeatures, trainTargets, validationTargets = train_test_split(
        trainValidationFeatures,
        trainValidationTargets,
        test_size=relativeValidationSize,
        random_state=config.randomState,
        stratify=trainValidationTargets
    )

    return trainFeatures, validationFeatures, testFeatures, trainTargets, validationTargets, testTargets


def createTensorDataset(featureMatrix: np.ndarray, targetVector: np.ndarray) -> TensorDataset:

    featureTensor = torch.tensor(featureMatrix, dtype=torch.float32)
    targetTensor = torch.tensor(targetVector, dtype=torch.float32).view(-1, 1)

    return TensorDataset(featureTensor, targetTensor)


def createDataLoaders(trainFeatures: np.ndarray, validationFeatures: np.ndarray, testFeatures: np.ndarray, trainTargets: np.ndarray, validationTargets: np.ndarray, testTargets: np.ndarray, batchSize: int) -> Tuple[DataLoader, DataLoader, DataLoader]:

    trainDataset = createTensorDataset(trainFeatures, trainTargets)
    validationDataset = createTensorDataset(validationFeatures, validationTargets)
    testDataset = createTensorDataset(testFeatures, testTargets)

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    validationLoader = DataLoader(validationDataset, batch_size=batchSize, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    return trainLoader, validationLoader, testLoader


def trainOneEpoch(model: nn.Module, dataLoader: DataLoader, lossFunction: nn.Module, optimizer: torch.optim.Optimizer) -> float:

    model.train()
    totalLoss = 0.0

    for inputBatch, targetBatch in dataLoader:
        inputBatch = inputBatch.to(device)
        targetBatch = targetBatch.to(device)

        optimizer.zero_grad()

        logits = model(inputBatch)
        loss = lossFunction(logits, targetBatch)

        loss.backward()
        optimizer.step()

        totalLoss += loss.item() * inputBatch.size(0)

    return totalLoss / len(dataLoader.dataset)


def evaluateModel(model: nn.Module, dataLoader: DataLoader,lossFunction: nn.Module) -> Dict[str, float]:

    model.eval()

    totalLoss = 0.0
    actualLabels = []
    predictedLabels = []

    with torch.no_grad():
        for inputBatch, targetBatch in dataLoader:
            inputBatch = inputBatch.to(device)
            targetBatch = targetBatch.to(device)

            logits = model(inputBatch)
            loss = lossFunction(logits, targetBatch)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            totalLoss += loss.item() * inputBatch.size(0)

            actualLabels.extend(targetBatch.cpu().numpy().flatten())
            predictedLabels.extend(predictions.cpu().numpy().flatten())

    return {
        "loss": totalLoss / len(dataLoader.dataset),
        "accuracy": accuracy_score(actualLabels, predictedLabels),
        "precision": precision_score(actualLabels, predictedLabels, zero_division=0),
        "recall": recall_score(actualLabels, predictedLabels, zero_division=0),
    }


def trainModel(model: nn.Module, trainLoader: DataLoader, validationLoader: DataLoader, config: TrainingConfig, modelName: str):

    lossFunction = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learningRate)

    bestValidationLoss = float("inf")
    bestModelWeights = copy.deepcopy(model.state_dict())
    patienceCounter = 0

    epochResults = []

    for epoch in range(1, config.maxEpochs + 1):
        trainLoss = trainOneEpoch(model, trainLoader, lossFunction, optimizer)
        validationMetrics = evaluateModel(model, validationLoader, lossFunction)

        epochResult = {
            "epoch": epoch,
            "trainLoss": trainLoss,
            "validationLoss": validationMetrics["loss"],
            "validationAccuracy": validationMetrics["accuracy"],
            "validationPrecision": validationMetrics["precision"],
            "validationRecall": validationMetrics["recall"],
        }

        epochResults.append(epochResult)

        print(
            f"[{modelName}] Epoch {epoch:02d}/{config.maxEpochs} | "
            f"Training Loss: {trainLoss:.4f} | "
            f"Validation Loss: {validationMetrics['loss']:.4f} | "
            f"Accuracy: {validationMetrics['accuracy']:.4f} | "
            f"Precision: {validationMetrics['precision']:.4f} | "
            f"Recall: {validationMetrics['recall']:.4f}"
        )

        if validationMetrics["loss"] < bestValidationLoss:
            bestValidationLoss = validationMetrics["loss"]
            bestModelWeights = copy.deepcopy(model.state_dict())
            patienceCounter = 0
        else:
            patienceCounter += 1

            if patienceCounter >= config.patience:
                print(f"Early stop")
                break

    model.load_state_dict(bestModelWeights)

    return model, epochResults


def runExperiment(model: nn.Module, modelName: str, dataLoaders, config: TrainingConfig):

    trainLoader, validationLoader, testLoader = dataLoaders

    model = model.to(device)

    model, epochResults = trainModel(model=model, trainLoader=trainLoader, validationLoader=validationLoader, config=config, modelName=modelName)

    lossFunction = nn.BCEWithLogitsLoss()
    testMetrics = evaluateModel(model, testLoader, lossFunction)

    print(f"\nResults:")
    print(f"Accuracy : {testMetrics['accuracy']:.4f}")
    print(f"Precision: {testMetrics['precision']:.4f}")
    print(f"Recall   : {testMetrics['recall']:.4f}")
    print()

    return {
        "epochResults": epochResults,
        "testResults": testMetrics
    }


def main():
    featureMatrix, targetVector = loadData(config.dataPath)

    (trainFeatures, validationFeatures, testFeatures, trainTargets, validationTargets, testTargets) = prepareSplits(featureMatrix, targetVector, config)

    standardScaler = StandardScaler()

    trainFeatures = standardScaler.fit_transform(trainFeatures)
    validationFeatures = standardScaler.transform(validationFeatures)
    testFeatures = standardScaler.transform(testFeatures)

    dataLoaders = createDataLoaders(trainFeatures=trainFeatures, validationFeatures=validationFeatures, testFeatures=testFeatures, trainTargets=trainTargets, validationTargets=validationTargets, testTargets=testTargets, batchSize=config.batchSize)

    inputDimension = trainFeatures.shape[1]

    baselineModel = Baseline(inputDimension=inputDimension, dropout=config.baselineDropout)

    improvedModel = Improved(inputDimension=inputDimension, dropout=config.improvedDropout)

    baselineResults = runExperiment(model=baselineModel, modelName="Baseline", dataLoaders=dataLoaders, config=config)

    improvedResults = runExperiment(model=improvedModel, modelName="Improved", dataLoaders=dataLoaders, config=config)

    print("Baseline vs. Improved")

    print(
        f"Baseline -> "
        f"Accuracy: {baselineResults['testResults']['accuracy']:.4f}, "
        f"Precision: {baselineResults['testResults']['precision']:.4f}, "
        f"Recall: {baselineResults['testResults']['recall']:.4f}"
    )

    print(
        f"Improved -> "
        f"Accuracy: {improvedResults['testResults']['accuracy']:.4f}, "
        f"Precision: {improvedResults['testResults']['precision']:.4f}, "
        f"Recall: {improvedResults['testResults']['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
