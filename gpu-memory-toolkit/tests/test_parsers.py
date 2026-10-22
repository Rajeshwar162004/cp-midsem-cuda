import unittest

from parser.ncu_parser import parse_ncu_rows
from parser.ptxas_parser import parse_ptxas_output
from profiler.ncu_runner import _extract_csv_payload, _parse_ncu_csv


class NcuParserTests(unittest.TestCase):
    def test_metric_rows_are_grouped_and_duration_is_normalized(self):
        csv_text = '''"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"1","kernelA","gpu__time_duration.sum","nsecond","2500"\n"1","kernelA","l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum","sector","128"\n"1","kernelA","l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum","request","32"\n'''
        profiles = parse_ncu_rows(_parse_ncu_csv(csv_text))
        self.assertEqual(1, len(profiles))
        self.assertEqual(2.5, profiles[0].duration_us)
        self.assertEqual(4.0, profiles[0].global_mem.load_sector_ratio)

    def test_wide_csv_uses_the_metric_unit_row(self):
        csv_text = '''"Kernel Name","gpu__time_duration.sum"\n"","msecond"\n"kernelA(int)","1.5"\n'''
        profile = parse_ncu_rows(_parse_ncu_csv(csv_text))[0]
        self.assertEqual("kernelA", profile.kernel_name)
        self.assertEqual(1500.0, profile.duration_us)

    def test_missing_metrics_are_reported(self):
        csv_text = '''"Kernel Name","Metric Name","Metric Value"\n"kernelA","gpu__time_duration.sum","100"\n'''
        profile = parse_ncu_rows(_parse_ncu_csv(csv_text))[0]
        self.assertIn("global_load_sectors", profile.missing_metrics)

    def test_ncu_progress_and_application_output_are_excluded_from_csv(self):
        mixed_output = '==PROF== Connected\napplication says hello\n"Kernel Name","Metric Name","Metric Value"\n"kernelA","Duration","10"\n'
        csv_text = _extract_csv_payload(mixed_output)
        self.assertEqual(1, len(_parse_ncu_csv(csv_text)))


class PtxasParserTests(unittest.TestCase):
    def test_spill_and_resource_usage_are_extracted(self):
        text = '''ptxas info    : Compiling entry function '_Z18bad_register_spillPKfPfi' for 'sm_89'\nptxas info    : Function properties for _Z18bad_register_spillPKfPfi\n    512 bytes stack frame, 64 bytes spill stores, 32 bytes spill loads\nptxas info    : Used 96 registers, 2048 bytes smem, 372 bytes cmem[0]\n'''
        report = parse_ptxas_output(text)
        self.assertEqual(1, len(report.kernels))
        kernel = report.kernels[0]
        self.assertTrue(kernel.has_spilling)
        self.assertEqual(96, kernel.registers)
        self.assertEqual(2048, kernel.shared_mem_bytes)
        self.assertEqual(kernel, report.get_kernel("bad_register_spill(float*)"))


if __name__ == "__main__":
    unittest.main()
