import java.io.{File, FileOutputStream}
import java.nio.channels.FileChannel
import java.nio.file.{Paths, StandardOpenOption}

import com.github.vumaasha.ml2npy.Ml2NpyCSR
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.scalatest.FlatSpec

import sys.process._
/**
  * Created by fermat on 10/1/17.
  */
class Ml2NpyCSRSpec extends FlatSpec{
    val nosetestspath="/home/fermat/anaconda2/bin/nosetests "
    val pathToTest = "/home/fermat/workspace/ml2npy/src/test/python/Npytest.py:"

    "ML2NpyFile" should "Convert to CSR matrix" in {

      val csrGen = new Ml2NpyCSR
      val data: Seq[Vector] = Seq(
        new SparseVector(3, Array(0), Array(0.1)),
        new SparseVector(3, Array(1), Array(0.2)),
        new SparseVector(3, Array(2), Array(0.3))
      )
      val labels = Seq(
        new DenseVector(Array(0, 1)),
        new DenseVector(Array(1, 0)),
        new DenseVector(Array(1, 0))
      )
      data.zip(labels).foreach(tup => csrGen.addRecord(tup._1, tup._2))
      val fos = new FileOutputStream(new File("/tmp/data.npz"))
      fos.write(csrGen.getBytes)
      fos.close()

      val command=nosetestspath + pathToTest+"test_5"
      val response=command.!
      assert(response==0)
    }
}
