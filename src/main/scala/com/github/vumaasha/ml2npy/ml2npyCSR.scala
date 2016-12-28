package com.github.vumaasha.ml2npy

import java.io.{ByteArrayOutputStream, File, FileOutputStream}
import java.nio.ByteBuffer
import java.util.zip.{ZipEntry, ZipOutputStream}

import org.apache.spark.ml.linalg.SparseVector

class ml2npyCSR(data: Seq[Float], indices: Seq[Int], indexPointers: Seq[Int], rows: Int, columns: Int) {
  val dataB: ByteBuffer = NpyFile[Float].addElements(data)
  val indicesB: ByteBuffer = NpyFile[Int].addElements(indices)
  val indexPointersB: ByteBuffer = NpyFile[Int].addElements(indexPointers)
  val shapeB: ByteBuffer = NpyFile[Int].addElements(Array(rows,columns))

  val zipOut = {
    val bos = new ByteArrayOutputStream()
    val zos = new ZipOutputStream(bos)
    def addEntry(file: String, bytes: Array[Byte]): Unit = {
      val entry = new ZipEntry(s"$file.npy")
      zos.putNextEntry(entry)
      zos.write(bytes)
      zos.closeEntry()
    }
    addEntry("data", dataB.array())
    addEntry("indices", indicesB.array())
    addEntry("indptr", indexPointersB.array())
    addEntry("shape", shapeB.array())
    zos.close()
    bos
  }

}

object ml2npyCSR {
  def apply(data: Seq[Float], indices: Seq[Int], indexPointers: Seq[Int], rows: Int, columns: Int): ml2npyCSR = new ml2npyCSR(data, indices, indexPointers, rows, columns)

  def apply(vectors: Seq[SparseVector]): ml2npyCSR = {
    val indices = vectors.flatMap(_.indices)
    val values: Seq[Float] = vectors.flatMap(_.values).map(_.toFloat)
    val indPtr = vectors.map(_.numActives).toArray.scanLeft(0)(_ + _)
    val rows = vectors.length
    val columns = vectors.map(_.size).max
    new ml2npyCSR(values, indices, indPtr, rows, columns)
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

    val csr = new ml2npyCSR(Seq(0.1f, 0.3f, 0.5f), Seq(1, 1, 0), Seq(0, 1, 2, 3), 3, 2)
    val fos = new FileOutputStream(new File("/tmp/data.npz"))
    csr.zipOut.writeTo(fos)
    csr.zipOut.close()
    fos.close()
  }
}

object SparseVectorTester {
  def main(args: Array[String]): Unit = {
    val x = Seq(new SparseVector(4, Array(0, 1), Array(0.3, 0.5)), new SparseVector(4, Array(1, 3), Array(0.3, 0.5)), new SparseVector(4, Array(1, 2), Array(0.3, 0.5)))
    val csr = ml2npyCSR(x)
    val fos = new FileOutputStream(new File("/tmp/csr.npz"))
    csr.zipOut.writeTo(fos)
    csr.zipOut.close()
    fos.close()

  }
}
