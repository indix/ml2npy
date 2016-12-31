package com.github.vumaasha.ml2npy

import java.io.{ByteArrayOutputStream, File, FileOutputStream}
import java.nio.ByteBuffer
import java.util.zip.{ZipEntry, ZipOutputStream}

import org.apache.hadoop.mapreduce.{RecordWriter, TaskAttemptContext}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by vumaasha on 25/12/16.
  */

class ml2npyCSR(data: Seq[Float],
                indices: Seq[Int],
                indexPointers: Seq[Int],
                rows: Int, columns: Int,
                labels: Seq[Short],
                numLabels: Int = 1) {

  require(indexPointers.length - 1 == rows)

  val dataB: ByteBuffer = NpyFile[Float].addElements(data)
  val indicesB: ByteBuffer = NpyFile[Int].addElements(indices)
  val indexPointersB: ByteBuffer = NpyFile[Int].addElements(indexPointers)
  val dataShape: mutable.WrappedArray[Int] = Array(rows, columns)
  val dataShapeB: ByteBuffer = NpyFile[Int].addElements(dataShape)
  val labelsB: ByteBuffer = NpyFile[Short].addElements(labels)
  val labelsShape: mutable.WrappedArray[Int] = Array(rows, numLabels)
  val labelsShapeB: ByteBuffer = NpyFile[Int].addElements(labelsShape)


  private def addEntry(zos: ZipOutputStream, file: String, bytes: Array[Byte]): Unit = {
    val entry = new ZipEntry(s"$file.npy")
    zos.putNextEntry(entry)
    zos.write(bytes)
    zos.closeEntry()
  }

  val zipOut = {
    val bos = new ByteArrayOutputStream()
    val zos = new ZipOutputStream(bos)
    def addLabels(labels: Seq[Short], index: Int): Unit = {
      require(labels.length == rows)
      val labelsB = NpyFile[Short].addElements(labels)
      addEntry(zos, s"label_$index", labelsB.array())
    }
    addEntry(zos, "data", dataB.array())
    addEntry(zos, "indices", indicesB.array())
    addEntry(zos, "indptr", indexPointersB.array())
    addEntry(zos, "data_shape", dataShapeB.array())
    addEntry(zos, "labels", labelsB.array())
    addEntry(zos, "labels_shape", labelsShapeB.array())
    zos.close()
    bos
  }

}

class ml2npyCSRBuffer extends RecordWriter[Vector, Vector] {
  val data: ArrayBuffer[(Vector, Vector)] = ArrayBuffer.empty

  override def write(key: Vector, value: Vector): Unit = {
    val tuple: (Vector, Vector) = (key, value)
    data += tuple
  }

  override def close(context: TaskAttemptContext): Unit = {
    val csr = ml2npyCSR(data.iterator)
  }
}

object ml2npyCSR {
  def apply(data: Seq[Float], indices: Seq[Int], indexPointers: Seq[Int], rows: Int, columns: Int, labels: Seq[Short], numLabels: Int): ml2npyCSR =
    new ml2npyCSR(data, indices, indexPointers, rows, columns, labels, numLabels)

  def apply(iterator: Iterator[(Vector, Vector)]): ml2npyCSR = {
    val records = iterator.toSeq
    val vectors: Seq[SparseVector] = for {
      row <- records
      data = row._1
    } yield {
      data match {
        case SparseVector(sz, ic, vs) => new SparseVector(sz, ic, vs)
        case v => v.toSparse
      }
    }

    val indices = vectors.flatMap(_.indices)
    val values: Seq[Float] = vectors.flatMap(_.values).map(_.toFloat)
    val indPtr = vectors.map(_.numActives).toArray.scanLeft(0)(_ + _)
    val rows = vectors.length
    val columns = vectors.map(_.size).max

    val labels: Seq[Short] = for {
      row <- records
      labelRecord = row._2.toArray
      label <- labelRecord
    } yield label.toShort
    val numLabels = records.maxBy(x => x._2.numActives)._2.numActives

    require(labels.length == (records.length * numLabels), "All the label vectors should be of same size")

    new ml2npyCSR(values, indices, indPtr, rows, columns, labels, numLabels)
  }
}

object ml2npyCSRTester {
  def main(args: Array[String]): Unit = {
    /*
    To test the export start ipython and run following commands

    import numpy as np
    from scipy.sparse import csr_matrix
    loader = np.load('/tmp/test.npz')
    csr_matrix((loader['data'],loader['indices'],loader['indptr']),shape=loader['shape']).toarray()

    The imported matrix should contain the below values
    array([[ 0.        ,  0.1       ],
       [ 0.        ,  0.30000001],
       [ 0.5       ,  0.        ]], dtype=float32)
     */

    val labels = Seq(0, 1, 0).map(_.toShort)
    val csr = new ml2npyCSR(Seq(0.1f, 0.3f, 0.5f), Seq(1, 1, 0), Seq(0, 1, 2, 3), 3, 2, labels, 1)
    val fos = new FileOutputStream(new File("/tmp/data.npz"))
    csr.zipOut.writeTo(fos)
    csr.zipOut.close()
    fos.close()
  }
}

object SparseVectorTester {
  def main(args: Array[String]): Unit = {
    val x = Seq(
      (new SparseVector(4, Array(0, 1), Array(0.3, 0.7)), new DenseVector(Array(1, 0))),
      (new SparseVector(4, Array(1, 3), Array(0.3, 0.5)), new DenseVector(Array(2, 1))),
      (new SparseVector(4, Array(1, 2), Array(0.3, 0.5)), new DenseVector(Array(2, 1)))
    )
    val labels = Seq(1.toShort, 2.toShort, 0.toShort)
    val csr = ml2npyCSR(x.iterator)
    val fos = new FileOutputStream(new File("/tmp/csr.npz"))
    csr.zipOut.writeTo(fos)
    csr.zipOut.close()
    fos.close()

  }
}
