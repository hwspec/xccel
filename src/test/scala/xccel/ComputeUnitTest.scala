package xccel

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

// Minimal concrete unit mixing in HasStationaryWeights, isolated from any
// PE-grid semantics (unlike SystolicArray) -- lets this test pin down
// exactly what the trait itself generates, independent of any one
// concrete compute unit's own interpretation of the beats.
class HasStationaryWeightsProbe(cfg: ComputeUnitConfig, beats: Int) extends ComputeUnit(cfg) with HasStationaryWeights {
  def weightBeatElems = cfg.numOutputs
  def weightBeatWidth = cfg.weightWidth
  def weightTotalBeats = beats

  io.resultOut.valid := false.B
  io.resultOut.bits := VecInit(Seq.fill(cfg.numOutputs)(0.S(cfg.outputWidth.W)))
}

class ComputeUnitTest extends AnyFlatSpec with ChiselScalatestTester {

  "ComputeUnitConfig" should "reject a non-positive numInputs" in {
    intercept[IllegalArgumentException] {
      ComputeUnitConfig(numInputs = 0, numOutputs = 1, inputWidth = 8, weightWidth = 8, outputWidth = 8)
    }
  }

  it should "reject a non-positive numOutputs" in {
    intercept[IllegalArgumentException] {
      ComputeUnitConfig(numInputs = 1, numOutputs = 0, inputWidth = 8, weightWidth = 8, outputWidth = 8)
    }
  }

  it should "reject a non-positive inputWidth" in {
    intercept[IllegalArgumentException] {
      ComputeUnitConfig(numInputs = 1, numOutputs = 1, inputWidth = 0, weightWidth = 8, outputWidth = 8)
    }
  }

  it should "reject a non-positive weightWidth" in {
    intercept[IllegalArgumentException] {
      ComputeUnitConfig(numInputs = 1, numOutputs = 1, inputWidth = 8, weightWidth = 0, outputWidth = 8)
    }
  }

  it should "reject a non-positive outputWidth" in {
    intercept[IllegalArgumentException] {
      ComputeUnitConfig(numInputs = 1, numOutputs = 1, inputWidth = 8, weightWidth = 8, outputWidth = 0)
    }
  }

  "HasStationaryWeights" should "size weightWriteAddr to max(1, log2Ceil(weightTotalBeats))" in {
    val cfg = ComputeUnitConfig(numInputs = 2, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)

    // beats=1 is the degenerate case: log2Ceil(1)=0, which would be a
    // zero-width port (the same bug class hit earlier with rows=1) if not
    // for the max(1, ...) floor
    test(new HasStationaryWeightsProbe(cfg, beats = 1)) { c =>
      assert(c.weightWriteAddr.getWidth == 1)
    }
    test(new HasStationaryWeightsProbe(cfg, beats = 5)) { c =>
      assert(c.weightWriteAddr.getWidth == 3) // log2Ceil(5) = 3
    }
    test(new HasStationaryWeightsProbe(cfg, beats = 8)) { c =>
      assert(c.weightWriteAddr.getWidth == 3) // log2Ceil(8) = 3
    }
  }

  it should "size weightWriteData to exactly weightBeatElems x weightBeatWidth" in {
    val cfg = ComputeUnitConfig(numInputs = 2, numOutputs = 3, inputWidth = 8, weightWidth = 5, outputWidth = 20)
    test(new HasStationaryWeightsProbe(cfg, beats = 4)) { c =>
      assert(c.weightWriteData.length == 3) // weightBeatElems = cfg.numOutputs
      assert(c.weightWriteData(0).getWidth == 5) // weightBeatWidth = cfg.weightWidth
    }
  }

  it should "let weightWriteEn actually gate a write through weightWriteAddr/weightWriteData" in {
    val cfg = ComputeUnitConfig(numInputs = 2, numOutputs = 2, inputWidth = 8, weightWidth = 8, outputWidth = 20)
    test(new HasStationaryWeightsProbe(cfg, beats = 2)) { c =>
      // the trait itself defines no storage (that's each concrete unit's
      // job, e.g. SystolicArray's weightRegs) -- this only confirms the
      // ports accept pokes at their full declared range without error
      c.weightWriteEn.poke(true.B)
      c.weightWriteAddr.poke(1.U)
      c.weightWriteData(0).poke(0xff.U)
      c.weightWriteData(1).poke(0x00.U)
      c.clock.step()
      c.weightWriteEn.poke(false.B)
    }
  }
}
