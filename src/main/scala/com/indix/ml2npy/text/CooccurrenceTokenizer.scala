package com.indix.ml2npy.text

import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}

/**
  * Created by vumaasha on 4/1/17.
  */
class CooccurrenceTokenizer extends RegexTokenizer {
  protected override def createTransformFunc: (String) => Seq[String] = { input =>
    val stopWordSet = StopWordsRemover.loadDefaultStopWords("english").toSet
    val tokens:Array[String] = super.createTransformFunc(input).toSet.toArray
    val filteredTokens = tokens.filter(token => !stopWordSet.contains(token))
    val coocc = for {
      (tokenI: String, i: Int) <- filteredTokens.zipWithIndex
      (tokenJ: String, j: Int) <- filteredTokens.zipWithIndex if j > i
    } yield {
      val (t1: String, t2: String) = if (i < j) (tokenI, tokenJ) else (tokenJ, tokenI)
      s"${t1}_$t2"
    }
    filteredTokens   ++ coocc
  }
}
