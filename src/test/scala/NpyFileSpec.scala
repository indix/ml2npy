import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.nio.file.{Paths, StandardOpenOption}
import java.nio.{ByteBuffer, ByteOrder}
import sys.process._
import scala.collection.immutable.Range.Inclusive
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.scalatest.FlatSpec
import com.indix.ml2npy.NpyFile

/**
  * Created by fermat on 10/1/17.
  */
class NpyFileSpec extends FlatSpec{
  val nosetestspath="/home/fermat/anaconda2/bin/nosetests "
  val pathToTest = "/home/fermat/workspace/ml2npy/src/test/resources/Npytest.py:"
  "NpyFile" should "Convert a float array correctly" in {
    val content = NpyFile[Float].addElements(Seq(0.3f, 0.5f))
    val channel = FileChannel.open(Paths.get("/tmp/test.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    channel.write(content)
    channel.close()
    val command=nosetestspath + pathToTest+"test_0"
    val response=command.!
    assert(response==0)
  }

  "NpyFile" should "Convert a sequence of integers correctly" in {
    val intContent = NpyFile[Int].addElements(1 to 10)
    val intChannel = FileChannel.open(Paths.get("/tmp/inttest.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    intChannel.write(intContent)
    intChannel.close()
    val command=nosetestspath + pathToTest+"test_1"
    val response=command.!
    assert(response==0)
  }

  "NpyFile" should "Convert a sequence to non 1- indexed itegers correctly" in {
    val intContent2 = NpyFile[Int].addElements(2 to 20)
    val intChannel2 = FileChannel.open(Paths.get("/tmp/inttest2.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    intChannel2.write(intContent2)
    intChannel2.close()
    val command=nosetestspath + pathToTest+"test_2"
    val response=command.!
    assert(response==0)
  }

  "NpyFile" should "write bytes directly" in {
    val data: Inclusive = 1 to 10
    val intContent3 = NpyFile[Int]
    data.foreach(intContent3.addToBuffer)
    val bytes = intContent3.getBytes
    val intChannel3 = FileChannel.open(Paths.get("/tmp/inttest3.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    intChannel3.write(ByteBuffer.wrap(bytes))
    intChannel3.close()
    val command=nosetestspath + pathToTest+"test_3"
    val response=command.!
    assert(response==0)
  }

  "NpyFile" should "should write bytes directly in a sequence" in {
    val data2: Inclusive = 11 to 20
    val intContent4 = NpyFile[Int]
    data2.foreach(intContent4.addToBuffer)
    val bytes2 = intContent4.getBytes
    val intChannel4 = FileChannel.open(Paths.get("/tmp/inttest4.npy"), StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE)
    intChannel4.write(ByteBuffer.wrap(bytes2))
    intChannel4.close()
    val command=nosetestspath + pathToTest+"test_4"
    val response=command.!
    assert(response==0)
  }
}
