package com.indix.ml2npy.preprocessing

import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.util._


trait DocGenerator[T,V] {

  def main(args: Array[String]): Unit = {

    val inputPath: String = args(0)
    val outputPath: String = args(1)
    val numParts: Int = args(2).toInt

    val spark = SparkSession.builder().appName("Ml2Npy").getOrCreate()
    val trainingRecords =readRecords(inputPath)
    val sampledRecords=sampleRecords(trainingRecords)
    writeRecords(sampledRecords,outputPath,pipeline)
  }

  def pipeline: Pipeline

  def readRecords(inputPath:String):Dataset[T]

  def sampleRecords(trainingRecords:Dataset[T]):Dataset[V]

  def writeRecords(sampledRecords: Dataset[V],outputPath:String,pipeline:Pipeline)
}