#! /usr/bin/env python
# encoding: utf-8
#

APPNAME='denn'
VERSION='0.1.0'

top='.'
out='build'

import os, shutil
from waflib.ConfigSet import ConfigSet

#
# Check command line options
#
def options(opt):
        opt.load('compiler_cxx')

#
# Project configuration 
#
# (At the moment, we check for the presences of certain libraries, 
#  but we need to decide on the course of action when these libraries are not
#  present.)
#
def configure(cnf):
        cnf.check_waf_version(mini='1.6.7')
        cnf.load('compiler_cxx doxygen')
        cnf.env.append_unique('CXXFLAGS', ['-O2', '-g', 
                    '-std=c++0x', '-Wall','-mtune=native','-march=native'])
        cnf.check_cxx(lib=['m'], uselib_store='M')
        cnf.check_cxx(lib=['rt'], uselib_store='M')
        cnf.check_cxx(lib=['stdc++'], uselib_store='M')
        cnf.check_cxx(lib=['pthread'], uselib_store='M')
        cnf.define('APPNAME',APPNAME)
        cnf.define('VERSION',VERSION)
        cnf.write_config_header('config.h')
        print('→ configuring the project in ' + cnf.path.abspath())

#
# Build the project
#
def build(bld):
        srcs = bld.path.ant_glob('src/main/c++/**/*.cc',src='true',bld='true')
        bld(features='cxx cxxprogram',source=srcs,
            includes = ['.', 'src/main/c++'],
            target=APPNAME, use=['M'])

#
# Build a distribution
#
def dist(ctx):
        ctx.algo      = 'tar.bz2'
        ctx.excl      = '**/.waf-* **/*~ **/*.git **/*.swp **/.lock-* DataSets build junk'

#
# Build the Documentation
#
def docs(cnf):
        cfgFile = os.path.join(out,os.path.join('c4che','_cache.py'))
        cnf.env = ConfigSet(cfgFile)
        cmd='%s doc/doxy.cfg' % cnf.env.DOXYGEN
        print('Executing the command: %s' % cmd )
        cnf.exec_command(cmd)
