all: .venv | deps/vattention deps/libtorch
	# cd deps/vattention/sarathi-lean; ../../../.venv/bin/pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.4/
	cd deps/vattention/vattention; LIBTORCH_PATH=../../libtorch CPATH=/usr/local/cuda-12.4/targets/x86_64-linux/include:${CPATH} LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:${LD_LIBRARY_PATH} PATH=/usr/local/cuda-12.4/bin:${PATH} ../../../.venv/bin/python setup.py install

clean:
	rm -rf deps

.venv:
	python3.11 -m venv .venv
	.venv/bin/pip install -r requirements.txt

deps/libtorch: | deps/libtorch-shared-with-deps-2.4.0.cu121.zip
	unzip deps/libtorch-shared-with-deps-2.4.0.cu121.zip -d deps

deps/vattention: | deps
	git clone https://github.com/microsoft/vattention deps/vattention

deps/libtorch-shared-with-deps-2.4.0.cu121.zip: | deps
	wget -O deps/libtorch-shared-with-deps-2.4.0.cu121.zip https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.4.0%2Bcu121.zip

deps:
	mkdir -p deps


.PHONE: build clean
