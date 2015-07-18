#include "UserKernel.h"

extern "C" {
  #include "lua.h"
  #include "utils.h"
  #include "luaT.h"
}
#include "EasyCL.h"
#include "THClKernels.h"

#include <iostream>
#include <string>
#include <vector>
using namespace std;

static std::string getTensorInfoClSrc();

enum ClKernelDirection {
  input,
  output,
  inout
};

class ClKernel;

class ClKernelArg {
public:
  ClKernelDirection direction;
  string name;
  ClKernelArg(ClKernelDirection direction, string name) :
    direction(direction),
    name(name) {
  }
  virtual std::string asParameterString() const = 0;
  virtual void writeToKernel(lua_State *L, ClKernel *clKernel,THClKernels *k) = 0;
  virtual ~ClKernelArg() {
  }
  virtual std::string toString() const {
    return "ClKernelArg{name=" + name + "}";
  }
};

class ClKernelArgInt : public ClKernelArg {
public:
  ClKernelArgInt(ClKernelDirection direction, string name) :
      ClKernelArg(direction, name) {
    if(direction != input) {
      THError("ints can only be input parameters, not output, or inout");
    }
  }
  virtual std::string asParameterString() const {
    return "int " + name;
  }
  virtual void writeToKernel(lua_State *L, ClKernel *clKernel,THClKernels *k) {
    luaT_getfieldchecknumber(L, -1, name.c_str());
    int value = lua_tonumber(L, -1);
    lua_pop(L, 1);
    switch(direction) {
      case input:
        k->in(value);
        break;
      default:
        THError("ints can only be input parameters, not output, or inout");
    }
  }
  virtual std::string toString() const {
    return "ClKernelArgInt{name=" + name + "}";
  }
};

class ClKernelArgTensor : public ClKernelArg {
public:
  ClKernelArgTensor(ClKernelDirection direction, string name) :
      ClKernelArg(direction, name) {
  }
  virtual std::string asParameterString() const {
    string res = "";
    res += "global struct THClTensorInfoCl *" + name + "_info, ";
    res += "global float * " + name + "_data";
    return res;
  }
  virtual void writeToKernel(lua_State *L, ClKernel *clKernel,THClKernels *k) {
    luaT_getfieldcheckudata(L, -1, name.c_str(), "torch.ClTensor");
    THClTensor *value = (THClTensor *)luaT_checkudata(L, -1, "torch.ClTensor");
    if(value == 0) {
      THError("Tensor is null.  this is odd actually, raise an issue");
    }
    if(value->storage == 0) {
      if(direction != output){
        THError("Output tensor has no data");
        return;
      } else {
        // I guess we should resize here, somehow, in the future
        THError("resize not implemented yet.  It should be.  Please remind me to add this, eg raise an issue");
        return;
      }
    }
    if(value->storage->wrapper == 0) {
      THError("resize not implemented yet.  It should be.  Please remind me to add this, eg raise an issue");
      return;
    }
    cout << "tensor value numElements " << value->storage->wrapper->size() << endl;
    switch(direction) {
      case input:
        if(!value->storage->wrapper->isOnDevice()) {
          value->storage->wrapper->copyToDevice();
        }
        k->inv2(value);
        break;
      case output:
        if(!value->storage->wrapper->isOnDevice()) {
          value->storage->wrapper->createOnDevice();
        }
        k->outv2(value);
        break;
      case inout:
        if(!value->storage->wrapper->isOnDevice()) {
          value->storage->wrapper->copyToDevice();
        }
        k->inoutv2(value);
        break;
    }
    lua_pop(L, 1);
  }
  virtual std::string toString() const {
    return "ClKernelArgTensor{name=" + name + "}";
  }
};

class ClKernelArgFloat : public ClKernelArg {
public:
  ClKernelArgFloat(ClKernelDirection direction, string name) :
      ClKernelArg(direction, name) {
    if(direction != input) {
      THError("floats can only be input parameters, not output, or inout");
    }
  }
  virtual std::string asParameterString() const {
    return "float " + name;
  }
  virtual void writeToKernel(lua_State *L, ClKernel *clKernel,THClKernels *k) {
    luaT_getfieldchecknumber(L, -1, name.c_str());
    float value = lua_tonumber(L, -1);
    lua_pop(L, 1);
    cout << "float value: " << value << endl;
    switch(direction) {
      case input:
        k->in(value);
        break;
      default:
        THError("ints can only be input parameters, not output, or inout");
    }
  }
  virtual std::string toString() const {
    return "ClKernelArgFloat{name=" + name + "}";
  }
};
int getNumElementsFromTensorArg(lua_State *L, ClKernelArgTensor *arg) {
  THClState *state = cltorch_getstate(L);
  luaT_getfieldcheckudata(L, -1, arg->name.c_str(), "torch.ClTensor");
  THClTensor *tensor = (THClTensor *)luaT_checkudata(L, -1, "torch.ClTensor");  
  return THClTensor_nElement(state, tensor);
}
class ClKernel {
public:
  int refCount;
  string source;
  string extraSource;
  string kernelName;
  string generatedSource;
  CLKernel *kernel;
  vector< ClKernelArg * >args;
  ClKernel() {
    kernel = 0;
    kernelName = "user_kernel";
    extraSource = "";
  }
  ~ClKernel() {
    for(int i = 0; i < (int)args.size(); i++) {
      delete args[i];
    }
    args.clear();
  }
};
static void ClKernel_rawInit(ClKernel *self) {
  self->refCount = 1;
  self->source = "";
}
static void printStack(string name, lua_State *L) {
  int top = lua_gettop(L);
  cout << name << " top: " << top << endl;
  for(int i = 1; i <= top; i++ ) {
    cout << "  stack[" << i << "] type=" << lua_typename(L, lua_type(L, i)) << " " << lua_type(L, i) << endl;
  }
}
static void loadParameters(lua_State *L, ClKernelDirection direction, ClKernel *self) {
  // assume that stack top is a table iwth the parameters
  // each parameter comprises name as key, and type as string value
  // valid typenames: int, float, torch.ClTensor, maybe ClTensor as shorthand
  // iterate this table...
  lua_pushnil(L);
  int argIndex = (int)self->args.size();
  while(lua_next(L, -2) != 0) {
    string name = lua_tostring(L, -2);
    string paramType = lua_tostring(L, -1);
    if(paramType == "float") {
      ClKernelArg *arg = new ClKernelArgFloat(direction, name);
      self->args.push_back(arg);
    } else if(paramType == "int") {
      ClKernelArg *arg = new ClKernelArgInt(direction, name);
      self->args.push_back(arg);
    } else if(paramType == "torch.ClTensor" || paramType == "ClTensor") {
      ClKernelArg *arg = new ClKernelArgTensor(direction, name);
      self->args.push_back(arg);
    } else {
      THError("Unrecognized typename %s", paramType.c_str());
    }
    lua_pop(L, 1);
  }
}
static int ClKernel_new(lua_State *L) {
  ClKernel *self = (ClKernel*)THAlloc(sizeof(ClKernel));
  self = new(self) ClKernel();
  ClKernel_rawInit(self);

  if(lua_type(L, 1) == LUA_TTABLE) {
    lua_pushnil(L);
    while(lua_next(L, -2) != 0) {
      string key = lua_tostring(L, -2);
      if(key == "input") {
        loadParameters(L, ClKernelDirection::input, self);
      } else if( key == "output") {
        loadParameters(L, ClKernelDirection::output, self);
      } else if( key == "inout") {
        loadParameters(L, ClKernelDirection::inout, self);
      } else if( key == "name") {
        self->kernelName = lua_tostring(L, -1); // probably should use luaT for this
      } else if( key == "src") {
        self->source = lua_tostring(L, -1);
      } else if( key == "funcs") {
        self->extraSource = lua_tostring(L, -1);
      } else {
        THError("Parameter %s not recognized", key.c_str());
      }
      lua_pop(L, 1);
    }

    if(self->source == "") {
      THError("Missing parameter src, or was empty");
    }

    THClState *state = cltorch_getstate(L);
    EasyCL *cl = THClState_getClv2(state, state->currentDevice);
    string kernelName = "user_kernel";  // can override by param, in future
    string generatedSource = "";
    generatedSource += easycl::replaceGlobal(getTensorInfoClSrc(), "{{MAX_CLTORCH_DIMS}}", easycl::toString(MAX_CLTORCH_DIMS)) + "\n";
    generatedSource += self->extraSource + "\n";
    generatedSource += "kernel void " + kernelName + "(\n";
    for(int i = 0; i < (int)self->args.size(); i++) {
      if(i > 0) {
        generatedSource += ",\n";
      }
      generatedSource += "    " + self->args[i]->asParameterString();
    }
    generatedSource += "\n) {\n";   // probalby should use ostringstream for this really, for speed
    generatedSource += self->source + "\n";
    generatedSource += "}\n";
    self->generatedSource = generatedSource;
    self->kernelName = kernelName;
    self->kernel = cl->buildKernelFromString(generatedSource, kernelName, "", "user_kernel");
  } else {
    THError("First parameter to torch.ClKernel should be a table");
  }

  luaT_pushudata(L, self, "torch.ClKernel");
  return 1;
}
static int ClKernel_free(lua_State *L) {
  ClKernel *self = (ClKernel*)THAlloc(sizeof(ClKernel));
  if(!self) {
    return 0;
  }
  if(THAtomicDecrementRef(&self->refCount))
  {
    delete self->kernel;
    self->~ClKernel();
    THFree(self);
  }
  return 0;
}
static int ClKernel_factory(lua_State *L) {
  THError("ClKernel_factory not implemented");
  return 0;
}
static int ClKernel_print(lua_State *L) {
  ClKernel *self = (ClKernel *)luaT_checkudata(L, 1, "torch.ClKernel");
  cout << "source=" << self->source << endl;
  return 0;
}
static int ClKernel_run(lua_State *L) {
  THClState *state = cltorch_getstate(L);
  if(lua_type(L, 2) != LUA_TTABLE) {
    THError("run method expects one parameter: a table, with named arg values in");
    return 0;
  }
  // now we can assume we have a table :-)

  ClKernel *self = (ClKernel *)luaT_checkudata(L, 1, "torch.ClKernel");
  try {
    THClKernels k(state, self->kernel);
    
    int numElements = -1;
    for(int i = 0; i < (int)self->args.size(); i++) {
      // what we need to do here is:
      // - loop through each parameter object
      // - for each parameter object, get the value from the passed in parameters, which
      //   were hopefully passed into this method, as values in a table
      // - and then write this value to the kernel
      // presumably, we can get each arg to extract itself from the table, as long as
      // the table is top of the stack
      // we can ignore extra values in the table for now
      // what we do is, throw error on missing values
      self->args[i]->writeToKernel(L, self, &k);
      if(numElements == -1 && self->args[i]->direction != input && dynamic_cast< ClKernelArgTensor *>( self->args[i] ) != 0) {
        numElements = getNumElementsFromTensorArg(L, dynamic_cast< ClKernelArgTensor *>(self->args[i]));
      }
    }
    if(numElements == -1) {
      THError("Must provide at least one output, or inout, ClTensor");
    }
    int workgroupSize = 64;  // should make this an option
    int numWorkgroups = (numElements + workgroupSize - 1) / workgroupSize;
    int globalSize = workgroupSize * numWorkgroups;
    self->kernel->run_1d(globalSize, workgroupSize);
  } catch(runtime_error &e) {
    THError("Error: %s", e.what());
  }
  return 0;
}
static const struct luaL_Reg ClKernel_funcs [] = {
  {"print", ClKernel_print},
  {"run", ClKernel_run},
  {0,0}
};
void cltorch_UserKernel_init(lua_State *L)
{
  luaT_newmetatable(L, "torch.ClKernel", NULL,
                    ClKernel_new, ClKernel_free, ClKernel_factory);
  luaL_setfuncs(L, ClKernel_funcs, 0);
  lua_pop(L, 1);
}
static std::string getTensorInfoClSrc() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "src/lib/THClTensorInfoCl.cl" )
  // ]]]
  // generated using cog, from src/lib/THClTensorInfoCl.cl:
  const char * kernelSource =  
  "typedef struct THClTensorInfoCl {\n" 
  "  unsigned int sizes[{{MAX_CLTORCH_DIMS}}];\n" 
  "  unsigned int strides[{{MAX_CLTORCH_DIMS}}];\n" 
  "  int offset;\n" 
  "  int dims;\n" 
  "} TensorInfoCl;\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

